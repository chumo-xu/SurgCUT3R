# 补丁嵌入(Patch Embedding)详细解释
# 本文件详细解释CUT3R项目中的补丁嵌入机制

"""
补丁嵌入是Vision Transformer的核心组件，负责将2D图像转换为1D序列。

核心思想：
将输入图像分割成不重叠的补丁(patches)，每个补丁被视为一个"词"，
然后通过线性投影将每个补丁映射到高维特征空间。

例如：
- 输入图像: 224×224×3
- 补丁大小: 16×16
- 补丁数量: (224/16)² = 196个补丁
- 每个补丁: 16×16×3 = 768维向量
- 嵌入维度: 768维(可配置)

CUT3R中的特殊处理：
1. ManyAR_PatchEmbed: 处理任意宽高比的图像
2. PatchEmbedDust3R: 标准的补丁嵌入
3. 支持不同输入通道数(RGB图像3通道，射线映射6通道)
"""

import torch
import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import PatchEmbed  # noqa


def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3):
    """
    补丁嵌入工厂函数
    
    根据指定的类名创建相应的补丁嵌入实例。
    
    参数:
        patch_embed_cls: 补丁嵌入类名 ("PatchEmbedDust3R" 或 "ManyAR_PatchEmbed")
        img_size: 输入图像尺寸
        patch_size: 补丁大小
        enc_embed_dim: 编码器嵌入维度
        in_chans: 输入通道数(RGB图像为3，射线映射为6)
        
    返回:
        补丁嵌入实例
    """
    assert patch_embed_cls in ["PatchEmbedDust3R", "ManyAR_PatchEmbed"]
    # 使用eval动态创建实例
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, in_chans, enc_embed_dim)
    return patch_embed


class PatchEmbedDust3R(PatchEmbed):
    """
    DUSt3R风格的补丁嵌入
    
    这是标准的补丁嵌入实现，适用于固定尺寸的图像。
    继承自基础的PatchEmbed类，添加了位置信息的获取。
    
    特点：
    - 要求输入图像尺寸是补丁大小的整数倍
    - 返回补丁特征和位置信息
    - 支持可选的归一化
    """
    
    def forward(self, x, **kw):
        """
        前向传播
        
        参数:
            x: 输入图像张量 (B, C, H, W)
            **kw: 其他关键字参数(未使用，保持接口一致性)
            
        返回:
            x: 补丁嵌入特征 (B, N, D) 或 (B, D, H', W')
            pos: 位置信息 (B, N, 2)
        """
        B, C, H, W = x.shape
        
        # 检查图像尺寸是否与补丁大小兼容
        assert (
            H % self.patch_size[0] == 0
        ), f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert (
            W % self.patch_size[1] == 0
        ), f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        
        # 1. 线性投影：将补丁转换为嵌入向量
        # 使用卷积实现，kernel_size=patch_size, stride=patch_size
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        # 2. 获取位置信息
        # position_getter返回每个补丁在原图中的2D坐标
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        
        # 3. 可选的展平操作
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        # 4. 归一化
        x = self.norm(x)
        
        return x, pos


class ManyAR_PatchEmbed(PatchEmbed):
    """
    支持任意宽高比的补丁嵌入
    
    这个类专门设计用于处理不同宽高比的图像，这在实际应用中很常见。
    同一批次中的所有图像具有相同的宽高比，但不同批次可以有不同的宽高比。
    
    核心创新：
    - 自动检测图像是横向还是纵向
    - 对纵向图像进行转置处理
    - 相应地调整位置编码
    
    使用场景：
    - 处理真实世界的图像数据
    - 支持不同的相机设置和图像格式
    - 提高模型对不同输入格式的鲁棒性
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        """
        初始化ManyAR_PatchEmbed
        
        参数与标准PatchEmbed相同，但增加了对任意宽高比的支持。
        """
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        """
        前向传播，支持任意宽高比
        
        参数:
            img: 输入图像张量 (B, C, H, W)
            true_shape: 真实形状信息 (B, 2)，包含每个图像的实际高度和宽度
            
        返回:
            x: 补丁嵌入特征 (B, N, embed_dim)
            pos: 位置信息 (B, N, 2)
        """
        B, C, H, W = img.shape

        # 1. 验证输入
        assert (
            H % self.patch_size[0] == 0
        ), f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert (
            W % self.patch_size[1] == 0
        ), f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (
            B,
            2,
        ), f"true_shape has the wrong shape={true_shape.shape}"

        # 2. 计算补丁网格尺寸
        W //= self.patch_size[0]  # 补丁网格宽度
        H //= self.patch_size[1]  # 补丁网格高度
        n_tokens = H * W          # 总补丁数

        # 3. 解析真实形状信息
        height, width = true_shape.T  # (B,) (B,)

        # 4. 判断图像方向
        # 注意：这里的实现假设所有图像都是横向的
        # 在实际的CUT3R实现中，这个逻辑可能更复杂
        is_landscape = torch.ones_like(width, dtype=torch.bool)
        is_portrait = ~is_landscape

        # 5. 初始化输出张量
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # 6. 处理横向图像
        if is_landscape.any():
            # 标准的补丁嵌入处理
            x[is_landscape] = (
                self.proj(img[is_landscape])  # 线性投影
                .permute(0, 2, 3, 1)          # BCHW -> BHWC
                .flatten(1, 2)                # BHW,C -> B(HW),C
                .float()
            )

        # 7. 处理纵向图像
        if is_portrait.any():
            # 对纵向图像进行转置处理
            x[is_portrait] = (
                self.proj(img[is_portrait].swapaxes(-1, -2))  # 转置后投影
                .permute(0, 2, 3, 1)                          # BCHW -> BHWC
                .flatten(1, 2)                                # BHW,C -> B(HW),C
                .float()
            )

        # 8. 设置位置编码
        # 横向图像使用标准的位置编码
        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        # 纵向图像使用转置的位置编码
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        # 9. 归一化
        x = self.norm(x)
        
        return x, pos


# 使用示例和详细说明
"""
补丁嵌入的工作原理详解：

1. 图像分割：
   输入图像 (224, 224, 3) -> 补丁 (14, 14, 16*16*3)
   每个16x16的补丁被展平为768维向量

2. 线性投影：
   768维补丁向量 -> embed_dim维嵌入向量
   这通过一个卷积层实现：Conv2d(3, embed_dim, kernel_size=16, stride=16)

3. 位置编码：
   为每个补丁分配2D位置坐标 (i, j)
   这些坐标后续会被转换为位置编码

4. 输出格式：
   - 特征: (B, N, D) 其中N=H*W/patch_size^2, D=embed_dim
   - 位置: (B, N, 2) 每个补丁的2D坐标

使用示例：

# 标准补丁嵌入
patch_embed = PatchEmbedDust3R(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768
)

img = torch.randn(2, 3, 224, 224)  # 批次大小2
features, positions = patch_embed(img)
print(f"Features shape: {features.shape}")    # [2, 196, 768]
print(f"Positions shape: {positions.shape}")  # [2, 196, 2]

# 任意宽高比补丁嵌入
many_ar_embed = ManyAR_PatchEmbed(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768
)

img = torch.randn(2, 3, 224, 224)
true_shape = torch.tensor([[224, 224], [224, 224]])
features, positions = many_ar_embed(img, true_shape)

关键优势：
1. 灵活性：支持不同的图像尺寸和宽高比
2. 效率：使用卷积实现，比循环更快
3. 兼容性：与标准Transformer架构完全兼容
4. 可扩展性：易于适应不同的输入通道数

在CUT3R中的应用：
- RGB图像处理：3通道输入
- 射线映射处理：6通道输入(3D坐标+方向)
- 多视角处理：每个视角独立进行补丁嵌入
- 位姿估计：位置信息用于几何约束
"""
