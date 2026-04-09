import torch

import torch.nn as nn

import torch.nn.functional as F

from .model import CauchyActivation



class ConvTransformBlock(nn.Module):
    """
    卷积变换块：包含两个卷积层、两个批归一化层和可选Dropout。
    
    结构：
        Conv2d(3x3) -> BatchNorm -> Activation -> Conv2d(3x3) -> BatchNorm -> Dropout
    
    这个块用于学习特征变换，是ResidualStage中的基本组件。
    
    参数：
        channels (int)：输入和输出通道数
        activation_mode (str)：激活函数模式，'cauchy'或'relu'。默认'cauchy'
        dropout (float)：Dropout概率。默认0.0（不使用Dropout）
    
    属性：
        conv1, conv2 (Conv2d)：两个3x3卷积层
        bn1, bn2 (BatchNorm2d)：两个批归一化层
        act1 (nn.Module)：第一个激活函数（在卷积1之后）
        dropout (nn.Module)：Dropout层
    """
    def __init__(self, channels, activation_mode="cauchy", dropout=0.0):
        super().__init__()
        # 第一个卷积分支：Conv -> BN -> Activation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = get_activation(activation_mode)

        # 第二个卷积分支：Conv -> BN
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # 可选的Dropout层
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        前向传播。
        
        参数：
            x (Tensor)：输入张量，形状为(B, C, H, W)
            
        返回：
            Tensor：输出张量，形状与输入相同
        """
        # 第一分支：卷积 -> 批归一化 -> 激活
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act1(h)

        # 第二分支：卷积 -> 批归一化 -> Dropout
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.dropout(h)
        return h



class ResidualStage(nn.Module):
    """
    残差阶段模块：包含多个ConvTransformBlock，支持两种残差连接模式。
    
    两种残差模式：
        1. "standard"：标准残差连接 h = h + transformed
        2. "cauchy"：Cauchy加权混合模式，mix = Cauchy加权的历史激活 + 当前变换
                    这种模式会保留历史层的特征信息，通过Cauchy距离加权
    
    关键特性：
        - 维护激活历史：记录每层的输入
        - Cauchy距离加权：使用可学习的lambda和d参数对历史做加权平均
        - 距离度量：距离 = 从当前层到历史层的索引差
    
    参数：
        channels (int)：通道数
        num_blocks (int)：该阶段包含的ConvTransformBlock数量
        activation_mode (str)：激活函数模式。默认'cauchy'
        residual_mode (str)：残差连接模式，'standard'或'cauchy'。默认'standard'
        dropout (float)：Dropout概率。默认0.0
        
    属性：
        residual_mode (str)：残差模式
        blocks (ModuleList)：包含的卷积块列表
        post_act (nn.Module)：阶段后的激活函数
        mix_raw_lambda (Parameter)：原始lambda参数（通过softplus激活），形状(num_blocks+1,)
        mix_raw_d (Parameter)：原始d参数（通过softplus激活），形状(num_blocks+1,)
    """
    def __init__(self, channels, num_blocks, activation_mode="cauchy", residual_mode="standard", dropout=0.0):
        super().__init__()
        if residual_mode not in {"standard", "cauchy"}:
            raise ValueError("residual_mode must be 'standard' or 'cauchy'")

        self.residual_mode = residual_mode
        # 创建num_blocks个卷积变换块
        self.blocks = nn.ModuleList(
            [ConvTransformBlock(channels, activation_mode=activation_mode, dropout=dropout) for _ in range(num_blocks)]
        )
        # 每个块后的激活函数
        self.post_act = get_activation(activation_mode)

        # Cauchy加权参数：num_blocks + 1是因为包括输入和所有块的输出
        self.mix_raw_lambda = nn.Parameter(torch.zeros(num_blocks + 1))
        self.mix_raw_d = nn.Parameter(torch.ones(num_blocks + 1))

    def _cauchy_mix(self, history, target_layer):
        """
        Cauchy距离加权混合历史激活。
        
        实现原理：
            1. 为每个历史层计算距离：distance = target_layer - history_index
            2. 计算Cauchy权重：weight[i] = lambda / (distance[i]^2 + d^2)
            3. 归一化权重使其求和为1
            4. 对历史激活进行加权平均
        
        参数：
            history (list)：历史激活列表，从旧到新排列
            target_layer (int)：当前目标层的索引
            
        返回：
            Tensor：加权混合后的激活，形状与history中单个元素相同
        """
        # 从参数中获取可学习的lambda和d值
        lam = F.softplus(self.mix_raw_lambda[target_layer]) + 1e-6
        d = F.softplus(self.mix_raw_d[target_layer]) + 1e-6
        count = len(history)

        # 计算历史中每个元素到当前层的距离
        # 距离从target_layer开始逐渐增大，最近的层距离最小
        distances = torch.arange(
            target_layer, target_layer - count, -1, device=history[0].device
        ).float()
        # Cauchy距离加权：距离越小权重越大
        weights = lam / (distances.pow(2) + d.pow(2))
        # 归一化权重
        # # weights = weights / (weights.sum() + 1e-8)  # 移除强制归一化以防止梯度消失

        # 将历史激活堆叠为5D张量：(序列长度, B, C, H, W)
        stacked = torch.stack(history, dim=0)
        # 按权重加权求和：(1, B, C, H, W) -> (B, C, H, W)
        return (weights[:, None, None, None, None] * stacked).sum(dim=0)

    def forward(self, x):
        """
        前向传播。
        
        参数：
            x (Tensor)：输入张量，形状为(B, C, H, W)
            
        返回：
            Tensor：经过所有块和残差连接后的输出
        """
        h = x
        # 初始化历史激活列表，从输入开始
        history = [h]

        for layer_idx, block in enumerate(self.blocks, start=1):
            # 获取变换后的特征
            transformed = block(h)
            
            if self.residual_mode == "standard":
                # 标准残差连接：新 = 旧 + 变换
                h = h + transformed
            else:
                # Cauchy模式：新 = 变换 + Cauchy加权历史混合
                h = transformed + self._cauchy_mix(history, layer_idx)
            
            # 应用后激活
            h = self.post_act(h)
            # 记录当前激活到历史中，用于后续层的Cauchy混合
            history.append(h)

        return h



class CauchyCNN(nn.Module):
    """
    Cauchy激活的CNN模型，用于图像分类（如CIFAR-100）。
    
    架构设计：
        Stem(卷积+BN+激活) 
        -> Stage1(2个Block) -> MaxPool
        -> Down1(通道扩展卷积) -> Stage2(2个Block) -> MaxPool
        -> Down2(通道扩展卷积) -> Stage3(2个Block)
        -> Head(全局平均池化 -> 线性分类层)
    
    通道维度变化：
        输入 (B, 3, H, W)
        -> Stem (B, base_channels, H, W)
        -> Stage1 (B, base_channels, H/2, W/2)  # 经过MaxPool
        -> Down1+Stage2 (B, 2*base_channels, H/4, W/4)  # 通道翻倍，再池化
        -> Down2+Stage3 (B, 4*base_channels, H/8, W/8)  # 通道再翻倍
        -> Head -> (B, num_classes)
    
    参数：
        num_classes (int)：分类类别数。默认10
        base_channels (int)：基础通道数，后续阶段通道数为此的倍数。默认64
        activation_mode (str)：激活函数模式，'cauchy'或'relu'。默认'cauchy'
        residual_mode (str)：残差连接模式，'standard'或'cauchy'。默认'standard'
    
    属性：
        stem (Sequential)：主干网络入口
        stage1, stage2, stage3 (Sequential)：三个残差阶段
        down1, down2 (Sequential)：通道提升的卷积
        head (Sequential)：分类头
    """
    def __init__(self, num_classes=10, base_channels=64, activation_mode="cauchy", residual_mode="standard"):
        super().__init__()
        
        # ===== Stem：初始卷积层 =====
        # 将3通道RGB图像转换为base_channels特征图
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            get_activation(activation_mode),
        )

        # ===== Stage 1：第一个残差阶段 =====
        # 2个residual块 + MaxPool（通道数不变，空间分辨率减半）
        self.stage1 = nn.Sequential(
            ResidualStage(
                base_channels,
                num_blocks=6,
                activation_mode=activation_mode,
                residual_mode=residual_mode,
                dropout=0.05,
            ),
            nn.MaxPool2d(2),  # H, W -> H/2, W/2
        )

        # ===== Down1：通道扩展层 =====
        # 将通道数从base_channels扩展到2*base_channels
        c2 = base_channels * 2
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            get_activation(activation_mode),
        )
        
        # ===== Stage 2：第二个残差阶段 =====
        # 2个residual块 + MaxPool（通道数 = 2*base_channels，空间再减半）
        self.stage2 = nn.Sequential(
            ResidualStage(
                c2,
                num_blocks=4,
                activation_mode=activation_mode,
                residual_mode=residual_mode,
                dropout=0.1,
            ),
            nn.MaxPool2d(2),  # H, W -> H/4, W/4
        )

        # ===== Down2：通道再次扩展层 =====
        # 将通道数从2*base_channels扩展到4*base_channels
        c3 = c2 * 2
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            get_activation(activation_mode),
        )
        
        # ===== Stage 3：第三个残差阶段 =====
        # 2个residual块（通道数 = 4*base_channels，无池化）
        self.stage3 = ResidualStage(
            c3,
            num_blocks=4,
            activation_mode=activation_mode,
            residual_mode=residual_mode,
            dropout=0.1,
        )

        # ===== Head：分类头 =====
        # 全局平均池化 -> 展平 -> 线性分类
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 任意分辨率 -> (1, 1)
            nn.Flatten(),  # (B, c3, 1, 1) -> (B, c3)
            nn.Linear(c3, num_classes),  # (B, c3) -> (B, num_classes)
        )

    def forward(self, x):
        """
        前向传播。
        
        参数：
            x (Tensor)：输入图像张量，形状为(B, 3, H, W)
            
        返回：
            Tensor：分类logits，形状为(B, num_classes)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x

def get_activation(mode):
    """
    根据指定模式返回相应的激活函数。
    
    参数：
        mode (str)：激活函数模式，可选值为 'cauchy' 或 'relu'
        
    返回：
        nn.Module：对应的激活函数实例
        
    异常：
        ValueError：如果mode不是'cauchy'或'relu'
    """
    if mode == "cauchy":
        return CauchyActivation()
    if mode == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError("activation_mode must be 'cauchy' or 'relu'")



class ImprovedCauchyCNN(nn.Module):
    """
    改进版CNN：保留Cauchy激活 & Cauchy残差，但改进整体架构。
    
    改进内容：
    1. 增加每个stage的块数（更深的网络）
    2. 移除dropout层（Cauchy参数需要清晰的梯度信号）
    3. 改进残差块的初始化
    4. 保留Cauchy激活和Cauchy残差机制
    
    架构：
    Stem → Stage1(8块) → MaxPool
         → Stage2(6块) → MaxPool  
         → Stage3(6块)
         → Head
    
    参数数量：约 7-8M（相比原来6.7M略多，但更深）
    """
    def __init__(self, num_classes=10, base_channels=64, activation_mode="cauchy", residual_mode="standard"):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            get_activation(activation_mode),
        )

        # Stage 1: 增加到8块(从6块)
        self.stage1 = nn.Sequential(
            ResidualStage(
                base_channels,
                num_blocks=8,  # ↑ 6 → 8
                activation_mode=activation_mode,
                residual_mode=residual_mode,
                dropout=0.0,  # ↓ 0.05 → 0.0
            ),
            nn.MaxPool2d(2),
        )

        c2 = base_channels * 2
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            get_activation(activation_mode),
        )
        
        # Stage 2: 增加到6块(从4块)
        self.stage2 = nn.Sequential(
            ResidualStage(
                c2,
                num_blocks=6,  # ↑ 4 → 6
                activation_mode=activation_mode,
                residual_mode=residual_mode,
                dropout=0.0,  # ↓ 0.1 → 0.0
            ),
            nn.MaxPool2d(2),
        )

        c3 = c2 * 2
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            get_activation(activation_mode),
        )
        
        # Stage 3: 增加到6块(从4块)
        self.stage3 = ResidualStage(
            c3,
            num_blocks=6,  # ↑ 4 → 6
            activation_mode=activation_mode,
            residual_mode=residual_mode,
            dropout=0.0,  # ↓ 0.1 → 0.0
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, num_classes),
        )
        
        # 改进初始化：对Conv层使用Kaiming初始化
        self._init_weights()

    def _init_weights(self):
        """更好的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x

class BottleneckBlock(nn.Module):
    """
    Bottleneck残差块 - ResNet-50/101的核心。
    
    结构:
        x → Conv1×1(C→C/4)→BN→ReLU
            Conv3×3(C/4)→BN→ReLU  
            Conv1×1(C/4→C)→BN
            → (x + 残差)
            → ReLU
    
    优点：
        1. 参数数量少：1×1卷积比3×3便宜4倍
        2. 梯度流通好：链条清晰，不易消失
        3. 性能优秀：ResNet-50在ImageNet达到76%+
    
    参数：
        in_channels (int)：输入通道数
        bottleneck_channels (int)：中间层通道数（通常in_channels//4）
        out_channels (int)：输出通道数（通常==in_channels）
        stride (int)：步长，用于下采样
    """
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()
        
        # 1×1卷积：通道降维 C→C/4
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3×3卷积：主计算，在低维空间进行
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 
                               kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 1×1卷积：通道升维 C/4→C
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 如果stride不是1或输入/输出通道不匹配，需要投影快捷连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)  # ✓ 激活在残差之后！
        
        return out



def extract_cauchy_params(model):
    """提取ResidualStage中的Cauchy参数用于训练过程追踪。"""
    params = {}
    for name, module in model.named_modules():
        if isinstance(module, ResidualStage):
            with torch.no_grad():
                lam = F.softplus(module.mix_raw_lambda).cpu().numpy()
                d = F.softplus(module.mix_raw_d).cpu().numpy()
            params[name] = {"lambda": lam, "d": d}
    return params
