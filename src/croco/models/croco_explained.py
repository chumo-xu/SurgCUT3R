# CroCo模型架构详细解释
# 本文件对CroCo (Cross-view Completion)模型进行详细解释

"""
CroCo是一个基于Transformer的跨视图补全模型，是CUT3R项目的基础架构。

核心思想：
CroCo使用掩码自编码器(Masked Autoencoder)的思想，通过一个视图的信息来重建另一个视图的掩码区域。
这种训练方式让模型学会理解不同视角之间的几何关系。

架构特点：
1. 编码器-解码器结构
2. 支持RoPE和余弦位置编码
3. 交叉注意力机制处理多视图信息
4. 灵活的掩码策略

数据流程：
输入图像1(掩码) + 输入图像2(完整) -> 编码器 -> 解码器 -> 重建图像1的掩码部分
"""

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from models.blocks import Block, DecoderBlock, PatchEmbed
from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
from models.masking import RandomMask

from transformers import PretrainedConfig
from transformers import PreTrainedModel


class CrocoConfig(PretrainedConfig):
    """
    CroCo模型的配置类
    
    定义了CroCo模型的所有超参数，包括：
    - 图像和补丁尺寸
    - 编码器和解码器的结构参数
    - 位置编码类型
    - 掩码比例等
    """
    model_type = "croco"

    def __init__(
        self,
        img_size=224,                    # 输入图像尺寸
        patch_size=16,                   # 补丁大小
        mask_ratio=0.9,                  # 掩码比例(90%的补丁被掩码)
        enc_embed_dim=768,               # 编码器特征维度
        enc_depth=12,                    # 编码器深度(Transformer层数)
        enc_num_heads=12,                # 编码器注意力头数
        dec_embed_dim=512,               # 解码器特征维度
        dec_depth=8,                     # 解码器深度(Transformer层数)
        dec_num_heads=16,                # 解码器注意力头数
        mlp_ratio=4,                     # MLP隐藏层倍数
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 归一化层
        norm_im2_in_dec=True,            # 是否在解码器中对第二张图像进行归一化
        pos_embed="cosine",              # 位置编码类型: cosine或RoPE100
    ):
        super().__init__()
        # 保存所有配置参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_im2_in_dec = norm_im2_in_dec
        self.pos_embed = pos_embed


class CroCoNet(PreTrainedModel):
    """
    CroCo网络的主要实现类
    
    这是一个基于Transformer的编码器-解码器模型，专门设计用于跨视图补全任务。
    模型接收两张图像：一张被掩码的图像和一张完整的参考图像，
    目标是重建被掩码图像的缺失部分。
    """

    config_class = CrocoConfig
    base_model_prefix = "croco"

    def __init__(self, config: CrocoConfig):
        """
        初始化CroCo网络
        
        参数:
            config: CrocoConfig配置对象
        """
        super().__init__(config)

        # 1. 设置补丁嵌入层
        # 将输入图像分割成补丁并嵌入到高维空间
        self._set_patch_embed(config.img_size, config.patch_size, config.enc_embed_dim)

        # 2. 设置掩码生成器
        # 用于随机掩码输入图像的补丁
        self._set_mask_generator(self.patch_embed.num_patches, config.mask_ratio)

        # 3. 设置位置编码
        self.pos_embed = config.pos_embed
        if config.pos_embed == "cosine":
            # 余弦位置编码：为每个补丁位置分配固定的位置编码
            enc_pos_embed = get_2d_sincos_pos_embed(
                config.enc_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer(
                "enc_pos_embed", torch.from_numpy(enc_pos_embed).float()
            )
            # 解码器的位置编码
            dec_pos_embed = get_2d_sincos_pos_embed(
                config.dec_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer(
                "dec_pos_embed", torch.from_numpy(dec_pos_embed).float()
            )
            self.rope = None  # 余弦编码不需要RoPE
        elif config.pos_embed.startswith("RoPE"):  # 例如 RoPE100
            # 旋转位置编码(RoPE)：动态计算位置编码
            self.enc_pos_embed = None  # RoPE不需要预计算的位置编码
            self.dec_pos_embed = None
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(config.pos_embed[len("RoPE") :])  # 提取频率参数
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + config.pos_embed)

        # 4. 设置编码器
        # 编码器负责处理输入图像并提取特征
        self.enc_depth = config.enc_depth
        self.enc_embed_dim = config.enc_embed_dim
        self.enc_blocks = nn.ModuleList(
            [
                Block(
                    config.enc_embed_dim,
                    config.enc_num_heads,
                    config.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=config.norm_layer,
                    rope=self.rope,
                )
                for i in range(config.enc_depth)
            ]
        )
        self.enc_norm = config.norm_layer(config.enc_embed_dim)

        # 5. 掩码令牌(在下游任务中通常不使用)
        self.mask_token = None

        # 6. 设置解码器
        # 解码器负责融合两个视图的信息并重建掩码区域
        self._set_decoder(
            config.enc_embed_dim,
            config.dec_embed_dim,
            config.dec_num_heads,
            config.dec_depth,
            config.mlp_ratio,
            config.norm_layer,
            config.norm_im2_in_dec,
        )

        # 7. 设置预测头
        # 将解码器输出转换为像素值
        self._set_prediction_head(config.dec_embed_dim, config.patch_size)

        # 8. 初始化权重
        self.initialize_weights()

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        """
        设置补丁嵌入层
        
        补丁嵌入将输入图像分割成不重叠的补丁，并将每个补丁嵌入到高维空间。
        例如：224x224的图像用16x16的补丁分割，得到14x14=196个补丁。
        """
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        """
        设置掩码生成器
        
        掩码生成器随机选择一定比例的补丁进行掩码，
        这些被掩码的补丁将不会输入到编码器中。
        """
        self.mask_generator = RandomMask(num_patches, mask_ratio)

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        """
        设置解码器
        
        解码器使用交叉注意力机制，将编码器的输出(来自掩码图像)
        与参考图像的特征进行融合，以重建掩码区域。
        """
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        
        # 编码器到解码器的特征转换
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        
        # 解码器的Transformer块
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,  # 是否对参考图像进行归一化
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        
        # 最终的归一化层
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_prediction_head(self, dec_embed_dim, patch_size):
        """
        设置预测头
        
        预测头将解码器的输出转换为像素值，
        输出维度是patch_size^2 * 3 (RGB三通道)。
        """
        self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)

    def initialize_weights(self):
        """
        初始化模型权重

        使用适当的初始化策略来确保训练的稳定性：
        - 补丁嵌入：使用特定的初始化方法
        - 线性层：使用Xavier均匀初始化
        - 层归一化：偏置为0，权重为1
        """
        # 初始化补丁嵌入
        self.patch_embed._init_weights()

        # 初始化掩码令牌(如果存在)
        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # 初始化线性层和层归一化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化的具体实现
        """
        if isinstance(m, nn.Linear):
            # 线性层使用Xavier均匀初始化
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 层归一化：偏置为0，权重为1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        编码单个图像

        这是CroCo模型的核心编码过程：
        1. 将图像分割成补丁并嵌入
        2. 添加位置编码
        3. 应用掩码(如果需要)
        4. 通过编码器Transformer块

        参数:
            image: 输入图像 (B, 3, H, W)
            do_mask: 是否应用掩码
            return_all_blocks: 是否返回所有块的输出

        返回:
            特征、位置信息、掩码
        """
        # 1. 补丁嵌入：将图像转换为补丁序列
        # x: (B, N, C) 其中N是补丁数量，C是嵌入维度
        # pos: (B, N, 2) 每个补丁的2D位置坐标
        x, pos = self.patch_embed(image)

        # 2. 添加位置编码
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]  # 广播到批次维度

        # 3. 应用掩码
        B, N, C = x.size()
        if do_mask:
            # 生成随机掩码
            masks = self.mask_generator(x)  # (B, N) 布尔张量
            # 只保留未被掩码的补丁
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            # 不应用掩码，所有补丁都可见
            masks = torch.zeros((B, N), dtype=bool)
            posvis = pos

        # 4. 通过编码器
        if return_all_blocks:
            # 返回所有块的输出(用于某些下游任务)
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])  # 最后一层进行归一化
            return out, pos, masks
        else:
            # 只返回最后一层的输出
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks

    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        解码器：融合两个视图的信息

        这是CroCo的核心创新：使用交叉注意力机制将两个视图的信息融合。

        参数:
            feat1: 第一个图像的编码特征(被掩码的图像)
            pos1: 第一个图像的位置信息
            masks1: 第一个图像的掩码
            feat2: 第二个图像的编码特征(参考图像)
            pos2: 第二个图像的位置信息
            return_all_blocks: 是否返回所有块的输出

        返回:
            解码后的特征
        """
        # 1. 特征维度转换：从编码器维度转换到解码器维度
        visf1 = self.decoder_embed(feat1)  # 可见的第一图像特征
        f2 = self.decoder_embed(feat2)     # 第二图像特征

        # 2. 处理掩码令牌
        B, Nenc, C = visf1.size()
        if masks1 is None:  # 下游任务：没有掩码
            f1_ = visf1
        else:  # 预训练：有掩码
            Ntotal = masks1.size(1)
            # 创建完整的特征序列，包括掩码位置
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)  # 填入可见特征

        # 3. 添加位置编码
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed

        # 4. 通过解码器块
        out = f1_
        out2 = f2
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                # 交叉注意力：out(查询) 关注 out2(键值)
                _out, out2 = blk(_out, out2, pos1, pos2)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2 = blk(out, out2, pos1, pos2)
            out = self.dec_norm(out)

        return out

    def patchify(self, imgs):
        """
        将图像转换为补丁表示

        这个函数将图像重新排列成补丁序列，用于计算重建损失。

        参数:
            imgs: 输入图像 (B, 3, H, W)

        返回:
            补丁序列 (B, L, patch_size^2 * 3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p  # 补丁网格的高度和宽度
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)  # 重新排列维度
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x, channels=3):
        """
        将补丁序列转换回图像

        这是patchify的逆操作，用于可视化重建结果。

        参数:
            x: 补丁序列 (N, L, patch_size^2 * channels)
            channels: 通道数

        返回:
            重建的图像 (N, channels, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))

        return imgs


# 使用示例和说明
"""
CroCo模型的典型使用流程：

1. 预训练阶段：
   - 输入两张相关的图像(如同一场景的不同视角)
   - 对第一张图像应用随机掩码
   - 使用第二张图像的信息来重建第一张图像的掩码区域
   - 通过这种方式学习跨视图的几何关系

2. 下游任务适应：
   - 移除掩码机制
   - 添加任务特定的头部网络
   - 在目标任务上进行微调

数据流程详解：
输入图像1(224x224x3) -> 补丁嵌入(196x768) -> 掩码(~20个可见补丁) -> 编码器 -> 解码器特征
输入图像2(224x224x3) -> 补丁嵌入(196x768) -> 编码器 -> 解码器参考特征
解码器: 融合两个特征 -> 重建掩码区域 -> 预测头 -> 输出(196x768)

关键创新点：
1. 交叉注意力：让模型学会利用其他视图的信息
2. 大掩码比例：强迫模型学习全局几何理解
3. 灵活的位置编码：支持不同的图像尺寸和宽高比
"""
