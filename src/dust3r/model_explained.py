# CUT3R模型架构详细解释
# 本文件对ARCroco3DStereo模型进行详细解释，帮助理解模型的结构和工作原理

"""
ARCroco3DStereo模型是CUT3R项目的核心模型，用于从多视角图像中重建3D场景。
该模型基于CroCo架构，并针对3D重建和位姿估计任务进行了扩展。

主要特点：
1. 基于Transformer的编码器-解码器架构
2. 支持多视角输入和处理
3. 可以同时输出3D点云、置信度、位姿等多种信息
4. 支持不同类型的头部网络(linear, dpt等)
5. 灵活的配置系统，可以适应不同的应用场景

模型继承关系：
ARCroco3DStereo -> CroCoNet -> PreTrainedModel
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    自定义输出类，用于ARCroco3DStereo模型
    
    这个类定义了模型的输出格式，包含两个主要字段：
    - ress: 模型的主要输出结果，通常包含3D点云、置信度等信息
    - views: 处理后的视图信息
    """
    ress: Optional[List[Any]] = None  # 模型的主要输出结果列表
    views: Optional[List[Any]] = None  # 处理后的视图信息列表


def load_model(model_path, device, verbose=True):
    """
    从预训练权重文件加载模型
    
    参数:
        model_path: 模型权重文件路径
        device: 运行设备(CPU或GPU)
        verbose: 是否打印详细信息
        
    返回:
        加载好权重的模型实例
    """
    if verbose:
        print("... loading model from", model_path)
    # 加载权重文件
    ckpt = torch.load(model_path, map_location="cpu")
    # 处理模型配置字符串，替换补丁嵌入类名
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR只用于处理非一致的宽高比
    # 确保landscape_only设置为False
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    # 使用eval动态创建模型实例
    net = eval(args)
    # 加载权重到模型
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    """
    ARCroco3DStereo模型的配置类
    
    这个类定义了模型的所有配置参数，包括:
    - 输出模式(3D点云、位姿等)
    - 头部类型(linear、dpt等)
    - 深度、置信度、位姿的处理模式
    - 网络结构参数(编码器/解码器维度、层数、头数等)
    """
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",           # 输出模式: pts3d, pts3d+pose, pts3d+desc等
        head_type="linear",            # 头部类型: linear或dpt
        depth_mode=("exp", -float("inf"), float("inf")),  # 深度处理模式
        conf_mode=("exp", 1, float("inf")),               # 置信度处理模式
        pose_mode=("exp", -float("inf"), float("inf")),   # 位姿处理模式
        freeze="none",                 # 冻结策略
        landscape_only=True,           # 是否只处理横向图像
        patch_embed_cls="PatchEmbedDust3R",  # 补丁嵌入类
        ray_enc_depth=2,               # 射线编码器深度
        state_size=324,                # 状态大小
        local_mem_size=256,            # 本地内存大小
        state_pe="2d",                 # 状态位置编码
        state_dec_num_heads=16,        # 状态解码器头数
        depth_head=False,              # 是否使用深度头
        rgb_head=False,                # 是否使用RGB头
        pose_conf_head=False,          # 是否使用位姿置信度头
        pose_head=False,               # 是否使用位姿头
        **croco_kwargs,                # CroCo模型的其他参数
    ):
        super().__init__()
        # 保存所有配置参数
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.local_mem_size = local_mem_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.croco_kwargs = croco_kwargs  # 存储CroCo模型的参数


class ARCroco3DStereo(CroCoNet):
    """
    ARCroco3DStereo是CUT3R项目的核心模型类
    
    这个模型继承自CroCoNet，并添加了3D重建和位姿估计的功能。
    它可以处理多视角输入，并输出3D点云、置信度、位姿等信息。
    """
    config_class = ARCroco3DStereoConfig  # 配置类
    base_model_prefix = "arcroco3dstereo"  # 模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点

    def __init__(self, config: ARCroco3DStereoConfig):
        """
        初始化ARCroco3DStereo模型
        
        参数:
            config: 模型配置对象
        """
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        # 填充默认参数
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        # 创建CroCo配置
        croco_cfg = CrocoConfig(**self.croco_args)
        # 调用父类初始化
        super().__init__(croco_cfg)
        
        # 创建射线映射编码器块
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        
        # 设置下游头部网络
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        # 设置冻结策略
        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        """
        从预训练模型加载

        这个方法支持两种加载方式：
        1. 从本地文件加载
        2. 从HuggingFace Hub加载
        """
        if os.path.isfile(pretrained_model_name_or_path):
            # 如果是本地文件，使用load_model函数
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                # 尝试从HuggingFace Hub加载
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        """
        设置补丁嵌入层

        这个方法创建两个补丁嵌入层：
        1. patch_embed: 用于处理RGB图像(3通道)
        2. patch_embed_ray_map: 用于处理射线映射(6通道)
        """
        # 创建RGB图像的补丁嵌入
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        # 创建射线映射的补丁嵌入(6通道：3D坐标xyz + 3D方向)
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        **kw,
    ):
        """
        设置下游任务的头部网络

        根据配置创建相应的头部网络，用于输出不同类型的结果：
        - 3D点云
        - 置信度
        - RGB颜色
        - 位姿信息
        """
        # 保存配置参数
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.landscape_only = landscape_only

        # 根据输出模式确定是否需要置信度
        has_conf = ("conf" in output_mode) or depth_head

        # 使用头部工厂创建相应的头部网络
        self.downstream_head1 = head_factory(
            head_type, output_mode, self, has_conf, depth_head, rgb_head, pose_conf_head, pose_head
        )

    def set_freeze(self, freeze):
        """
        设置模型的冻结策略

        参数:
            freeze: 冻结策略，可以是：
                - "none": 不冻结任何参数
                - "encoder": 只冻结编码器
                - "decoder": 只冻结解码器
                - 其他自定义策略
        """
        self.freeze = freeze
        to_be_frozen = []

        if freeze == "none":
            pass  # 不冻结任何参数
        elif freeze == "encoder":
            # 冻结编码器相关参数
            to_be_frozen = [self.patch_embed, self.enc_blocks, self.enc_norm]
            if hasattr(self, "enc_pos_embed") and self.enc_pos_embed is not None:
                to_be_frozen.append(self.enc_pos_embed)
        elif freeze == "decoder":
            # 冻结解码器相关参数
            to_be_frozen = [self.decoder_embed, self.dec_blocks, self.dec_norm]
            if hasattr(self, "dec_pos_embed") and self.dec_pos_embed is not None:
                to_be_frozen.append(self.dec_pos_embed)
        else:
            # 其他冻结策略
            raise NotImplementedError(f"Unknown freeze strategy: {freeze}")

        # 执行冻结
        for module in to_be_frozen:
            freeze_all_params(module)

    def _encode_image(self, image, true_shape, do_mask=False, return_all_blocks=False):
        """
        编码单个图像

        参数:
            image: 输入图像张量 (B, 3, H, W)
            true_shape: 图像的真实形状信息
            do_mask: 是否进行掩码处理
            return_all_blocks: 是否返回所有块的输出

        返回:
            编码后的特征、位置信息和掩码
        """
        # 将图像转换为补丁嵌入
        x, pos = self.patch_embed(image, true_shape)

        # 添加位置编码
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]

        # 应用掩码(如果需要)
        B, N, C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            masks = torch.zeros((B, N), dtype=bool)
            posvis = pos

        # 通过编码器块
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks

    def forward(self, views, ret_state=False):
        """
        模型的前向传播方法

        这是模型的主要入口点，处理多视角输入并返回结果。

        参数:
            views: 输入视图列表，每个视图包含图像和相关信息
            ret_state: 是否返回状态信息

        返回:
            ARCroco3DStereoOutput对象，包含处理结果
        """
        if ret_state:
            # 如果需要返回状态信息
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            # 正常前向传播
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def _forward_impl(self, views, ret_state=False):
        """
        前向传播的具体实现

        这个方法包含了模型的核心逻辑：
        1. 处理输入视图
        2. 编码图像特征
        3. 解码并生成输出

        数据流程：
        输入图像 -> 补丁嵌入 -> 编码器 -> 解码器 -> 头部网络 -> 输出结果
        """
        # 预处理输入视图
        views = self._preprocess_views(views)

        # 编码所有视图
        encoded_features = []
        for view in views:
            # 获取图像和真实形状
            img = view['img']
            true_shape = view.get('true_shape', None)

            # 编码图像
            feat, pos, mask = self._encode_image(
                img, true_shape, do_mask=False, return_all_blocks=True
            )
            encoded_features.append((feat, pos, mask))

        # 解码并生成输出
        ress = []
        for i, (feat, pos, mask) in enumerate(encoded_features):
            # 使用头部网络生成输出
            output = self.downstream_head1(feat, views[i]['img'].shape[-2:])
            ress.append(output)

        if ret_state:
            # 如果需要返回状态，这里应该包含状态相关的逻辑
            # 为了简化，这里返回空的状态参数
            state_args = []
            return ress, views, state_args
        else:
            return ress, views

    def _preprocess_views(self, views):
        """
        预处理输入视图

        这个方法对输入的视图进行标准化处理：
        1. 确保所有视图具有相同的格式
        2. 处理图像尺寸和形状信息
        3. 添加必要的元数据
        """
        processed_views = []

        for view in views:
            processed_view = {}

            # 复制基本信息
            for key in ['img', 'true_shape', 'idx', 'instance']:
                if key in view:
                    processed_view[key] = view[key]

            # 确保图像是正确的格式
            if 'img' in processed_view:
                img = processed_view['img']
                if img.dim() == 3:
                    img = img.unsqueeze(0)  # 添加批次维度
                processed_view['img'] = img

            # 如果没有真实形状信息，从图像中推断
            if 'true_shape' not in processed_view and 'img' in processed_view:
                img = processed_view['img']
                B, C, H, W = img.shape
                processed_view['true_shape'] = torch.tensor([[H, W]] * B)

            processed_views.append(processed_view)

        return processed_views


# 配置示例和使用说明
"""
使用示例：

# 1. 创建配置
config = ARCroco3DStereoConfig(
    state_size=768,                    # 状态大小
    state_pe='2d',                     # 2D位置编码
    pos_embed='RoPE100',               # RoPE位置编码，频率100
    rgb_head=True,                     # 启用RGB头
    pose_head=True,                    # 启用位姿头
    patch_embed_cls='ManyAR_PatchEmbed', # 使用ManyAR补丁嵌入
    img_size=(512, 512),               # 输入图像尺寸
    head_type='dpt',                   # 使用DPT头部
    output_mode='pts3d+pose',          # 输出3D点云和位姿
    depth_mode=('exp', -inf, inf),     # 深度模式：指数激活
    conf_mode=('exp', 1, inf),         # 置信度模式：指数激活，最小值1
    pose_mode=('exp', -inf, inf),      # 位姿模式：指数激活
    enc_embed_dim=1024,                # 编码器嵌入维度
    enc_depth=24,                      # 编码器深度(层数)
    enc_num_heads=16,                  # 编码器注意力头数
    dec_embed_dim=768,                 # 解码器嵌入维度
    dec_depth=12,                      # 解码器深度(层数)
    dec_num_heads=12,                  # 解码器注意力头数
    landscape_only=False               # 支持任意宽高比
)

# 2. 创建模型
model = ARCroco3DStereo(config)

# 3. 准备输入数据
views = [
    {
        'img': torch.randn(1, 3, 512, 512),  # 第一个视图
        'true_shape': torch.tensor([[512, 512]])
    },
    {
        'img': torch.randn(1, 3, 512, 512),  # 第二个视图
        'true_shape': torch.tensor([[512, 512]])
    }
]

# 4. 前向传播
output = model(views)
results = output.ress  # 获取结果列表

# 5. 解析输出
for i, result in enumerate(results):
    print(f"View {i} outputs:")
    print(f"  - 3D points: {result['pts3d_in_self_view'].shape}")
    print(f"  - Confidence: {result['conf_self'].shape}")
    if 'camera_pose' in result:
        print(f"  - Camera pose: {result['camera_pose'].shape}")
    if 'rgb' in result:
        print(f"  - RGB: {result['rgb'].shape}")
"""


# ============================================================================
# 补充缺失的重要组件和方法详细解释
# ============================================================================

@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    ARCroco3DStereo模型的自定义输出类

    这个类定义了模型的输出格式，继承自HuggingFace的ModelOutput基类。
    它确保输出具有一致的格式，便于后续处理和分析。

    属性:
        ress: 模型的主要输出结果列表，每个元素对应一个视角的预测结果
        views: 输入的视角数据，用于调试和可视化

    使用场景:
        - 标准化模型输出格式
        - 支持HuggingFace生态系统
        - 便于结果的序列化和反序列化
        - 提供一致的API接口
    """
    ress: Optional[List[Any]] = None  # 主要输出：每个视角的预测结果
    views: Optional[List[Any]] = None  # 输入视角：用于调试和后处理


def strip_module(state_dict):
    """
    移除state_dict中键名的'module.'前缀

    在分布式训练中，模型会被包装在DataParallel或DistributedDataParallel中，
    这会在所有参数名前添加'module.'前缀。这个函数用于移除这些前缀，
    使得模型可以正确加载预训练权重。

    参数:
        state_dict (dict): 原始的state_dict，可能包含'module.'前缀

    返回:
        OrderedDict: 移除'module.'前缀后的新state_dict

    使用场景:
        - 加载分布式训练保存的模型权重
        - 在单GPU环境中使用多GPU训练的模型
        - 模型权重的格式转换

    实现原理:
        遍历state_dict的所有键，检查是否以'module.'开头，
        如果是则移除前7个字符，否则保持原样。
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 如果键名以'module.'开头，则移除前7个字符('module.')
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    """
    从检查点文件加载预训练模型

    这个函数处理模型加载的复杂逻辑，包括配置解析、权重加载和设备转移。
    它特别处理了ManyAR_PatchEmbed到PatchEmbedDust3R的转换，以及landscape_only参数的设置。

    参数:
        model_path (str): 模型检查点文件路径
        device (str): 目标设备 ('cpu', 'cuda', etc.)
        verbose (bool): 是否打印详细信息

    返回:
        model: 加载完成的模型实例

    处理的关键问题:
        1. 配置字符串的解析和修正
        2. 补丁嵌入类名的转换
        3. landscape_only参数的兼容性处理
        4. 权重的严格/非严格加载

    兼容性处理:
        - ManyAR_PatchEmbed -> PatchEmbedDust3R: 处理不同版本的补丁嵌入
        - landscape_only参数: 确保向后兼容性
        - 权重键名: 处理分布式训练的权重格式
    """
    if verbose:
        print("... loading model from", model_path)

    # 加载检查点
    ckpt = torch.load(model_path, map_location="cpu")

    # 处理配置字符串：将ManyAR_PatchEmbed替换为PatchEmbedDust3R
    # ManyAR只用于宽高比不一致的情况
    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")

    # 处理landscape_only参数的兼容性
    if "landscape_only" not in args:
        # 如果配置中没有landscape_only参数，添加默认值False
        args = args[:-2] + ", landscape_only=False))"
    else:
        # 如果有landscape_only参数，确保设置为False
        args = args.replace(" ", "").replace("landscape_only=True", "landscape_only=False")

    # 确保landscape_only=False存在于配置中
    assert "landscape_only=False" in args

    if verbose:
        print(f"instantiating : {args}")

    # 动态创建模型实例
    net = eval(args)

    # 加载模型权重（非严格模式，允许部分权重不匹配）
    s = net.load_state_dict(ckpt["model"], strict=False)

    if verbose:
        print(s)  # 打印加载状态信息

    return net.to(device)


class LocalMemory(nn.Module):
    """
    局部记忆模块 - CUT3R的核心创新组件

    LocalMemory是一个可学习的记忆系统，用于存储和检索位姿相关的信息。
    它允许模型在处理视频序列时维护长期的空间记忆，这对于准确的位姿估计至关重要。

    核心思想:
        1. 维护一个可学习的记忆库，存储历史位姿信息
        2. 通过write操作更新记忆内容
        3. 通过read操作查询相关的历史信息
        4. 使用注意力机制进行记忆的读写操作

    架构设计:
        - 记忆库: 固定大小的可学习参数矩阵
        - 写入块: 用于更新记忆内容的Transformer块
        - 读取块: 用于查询记忆内容的Transformer块
        - 掩码令牌: 用于查询时的占位符

    应用场景:
        - 视频序列的位姿估计
        - 长期依赖关系的建模
        - 空间记忆的维护
        - 历史信息的检索

    参数:
        size: 记忆库的大小（记忆槽的数量）
        k_dim: 键（查询）的维度
        v_dim: 值（内容）的维度
        num_heads: 注意力头的数量
        depth: Transformer块的深度
        其他参数: 标准Transformer参数
    """

    def __init__(
        self,
        size,           # 记忆库大小
        k_dim,          # 键维度
        v_dim,          # 值维度
        num_heads,      # 注意力头数
        depth=2,        # 网络深度
        mlp_ratio=4.0,  # MLP扩展比例
        qkv_bias=False, # QKV偏置
        drop=0.0,       # Dropout率
        attn_drop=0.0,  # 注意力Dropout率
        drop_path=0.0,  # DropPath率
        act_layer=nn.GELU,      # 激活函数
        norm_layer=nn.LayerNorm, # 归一化层
        norm_mem=True,  # 是否对记忆进行归一化
        rope=None,      # 旋转位置编码
    ) -> None:
        super().__init__()

        self.v_dim = v_dim

        # 查询投影层：将输入特征投影到记忆空间
        self.proj_q = nn.Linear(k_dim, v_dim)

        # 掩码令牌：用于查询时的占位符
        # 初始化为小的随机值，允许模型学习合适的查询表示
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )

        # 记忆库：存储历史信息的可学习参数
        # 维度为 [1, size, 2*v_dim]，其中2*v_dim允许存储键值对
        # 前v_dim维存储键信息，后v_dim维存储值信息
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )

        # 写入块：用于更新记忆内容
        # 这些Transformer块决定如何将新信息整合到记忆中
        self.write_blocks = nn.ModuleList([
            DecoderBlock(
                2 * v_dim,      # 输入维度（键值对）
                num_heads,      # 注意力头数
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_mem=norm_mem,
                rope=rope,
            )
            for _ in range(depth)
        ])

        # 读取块：用于查询记忆内容
        # 这些Transformer块决定如何从记忆中检索相关信息
        self.read_blocks = nn.ModuleList([
            DecoderBlock(
                2 * v_dim,      # 输入维度（查询+掩码）
                num_heads,      # 注意力头数
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_mem=norm_mem,
                rope=rope,
            )
            for _ in range(depth)
        ])

    def update_mem(self, mem, feat_k, feat_v):
        """
        更新记忆内容

        这个方法将新的特征信息写入记忆库。它使用注意力机制来决定
        如何更新记忆内容，允许模型选择性地保留重要信息。

        参数:
            mem: [B, size, 2*C] 当前记忆状态
            feat_k: [B, 1, C] 新的键特征（用于查询）
            feat_v: [B, 1, C] 新的值特征（要存储的内容）

        返回:
            updated_mem: [B, size, 2*C] 更新后的记忆状态

        工作流程:
            1. 将键特征投影到记忆空间
            2. 将键值特征拼接
            3. 通过写入块更新记忆内容
            4. 返回更新后的记忆

        设计原理:
            - 使用注意力机制决定更新哪些记忆槽
            - 允许模型学习如何整合新旧信息
            - 保持记忆库的固定大小
            - 支持批量处理
        """
        # 投影键特征到记忆空间
        feat_k = self.proj_q(feat_k)  # [B, 1, C]

        # 拼接键值特征，形成完整的记忆条目
        feat = torch.cat([feat_k, feat_v], dim=-1)  # [B, 1, 2*C]

        # 通过写入块更新记忆
        # 每个块都可以修改记忆内容，实现渐进式更新
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)

        return mem

    def inquire(self, query, mem):
        """
        查询记忆内容

        这个方法根据查询特征从记忆库中检索相关信息。它使用注意力机制
        来找到最相关的记忆内容，并返回对应的值特征。

        参数:
            query: [B, 1, C] 查询特征
            mem: [B, size, 2*C] 记忆库状态

        返回:
            result: [B, 1, C] 查询结果（值特征）

        工作流程:
            1. 将查询投影到记忆空间
            2. 添加掩码令牌作为占位符
            3. 通过读取块查询记忆内容
            4. 返回查询到的值特征

        设计原理:
            - 使用注意力机制找到最相关的记忆
            - 掩码令牌作为输出的占位符
            - 支持软性的记忆检索（而非硬性索引）
            - 允许模型学习查询策略
        """
        # 投影查询特征到记忆空间
        x = self.proj_q(query)  # [B, 1, C]

        # 添加掩码令牌，形成查询序列
        # 掩码令牌将被更新为查询结果
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)

        # 通过读取块查询记忆
        # 每个块都可以细化查询结果
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)

        # 返回值部分（后半部分），即查询到的内容
        return x[..., -self.v_dim:]


# ============================================================================
# ARCroco3DStereo主模型类的关键方法解释
# ============================================================================

class ARCroco3DStereoExplained:
    """
    ARCroco3DStereo模型的关键方法详细解释

    这个类不是实际的模型实现，而是对原模型中关键方法的详细解释和说明。
    它帮助理解模型的工作原理和数据流。
    """

    def _encode_image_explained(self, image, true_shape):
        """
        图像编码方法解释

        这个方法将输入的RGB图像编码为特征表示。它是模型的第一个主要步骤，
        负责将像素级的图像信息转换为高级的语义特征。

        参数:
            image: [B, C, H, W] 输入RGB图像
            true_shape: [B, 2] 图像的真实尺寸（用于处理填充）

        返回:
            img_features: [List] 图像特征列表
            img_pos: [B, N, 2] 位置编码
            None: 占位符（保持接口一致性）

        工作流程:
            1. 补丁嵌入：将图像分割为补丁并嵌入
            2. 位置编码：添加空间位置信息
            3. Transformer编码：通过多层注意力提取特征
            4. 归一化：最终的特征归一化

        关键技术:
            - 补丁嵌入：将2D图像转换为1D序列
            - 位置编码：保持空间关系信息
            - 自注意力：捕获全局依赖关系
            - 梯度检查点：内存优化技术
        """
        # 1. 补丁嵌入：将图像分割为补丁并投影到特征空间
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # 2. 确保没有使用传统的位置嵌入（使用RoPE代替）
        assert self.enc_pos_embed is None

        # 3. 通过编码器块处理特征
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点节省内存
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                # 正常前向传播
                x = blk(x, pos)

        # 4. 最终归一化
        x = self.enc_norm(x)

        return [x], pos, None

    def _encode_ray_map_explained(self, ray_map, true_shape):
        """
        射线映射编码方法解释

        这个方法编码射线映射信息，射线映射包含了相机的几何信息，
        对于3D重建和位姿估计至关重要。

        参数:
            ray_map: [B, C, H, W] 射线映射（C=6，包含射线原点和方向）
            true_shape: [B, 2] 真实尺寸

        返回:
            ray_features: [List] 射线特征列表
            ray_pos: [B, N, 2] 位置编码
            None: 占位符

        射线映射的组成:
            - 前3个通道：射线原点 (ray origin)
            - 后3个通道：射线方向 (ray direction)

        应用意义:
            - 提供几何约束信息
            - 支持多视角几何推理
            - 增强3D理解能力
            - 改善位姿估计精度
        """
        # 1. 使用专门的射线映射补丁嵌入
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)

        # 2. 确保没有使用传统位置嵌入
        assert self.enc_pos_embed is None

        # 3. 通过专门的射线映射编码器块
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)

        # 4. 射线映射特征归一化
        x = self.enc_norm_ray_map(x)

        return [x], pos, None

    def _encode_state_explained(self, image_tokens, image_pos):
        """
        状态编码方法解释

        这个方法初始化模型的全局状态表示。状态是CUT3R的核心概念，
        它维护了跨视角和跨时间的全局信息。

        参数:
            image_tokens: [B, N, C] 图像特征令牌
            image_pos: [B, N, 2] 图像位置编码

        返回:
            state_feat: [B, state_size, C] 状态特征
            state_pos: [B, state_size, 2] 状态位置编码
            None: 占位符

        状态的作用:
            - 维护全局场景信息
            - 支持长序列处理
            - 提供跨视角的一致性
            - 存储历史信息

        位置编码策略:
            - '1d': 一维位置编码
            - '2d': 二维网格位置编码
            - 'none': 不使用位置编码
        """
        batch_size = image_tokens.shape[0]

        # 1. 获取状态特征：从可学习的寄存器令牌开始
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )

        # 2. 根据配置生成状态位置编码
        if self.state_pe == "1d":
            # 一维位置编码：每个状态令牌有相同的位置
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "2d":
            # 二维位置编码：将状态令牌排列为2D网格
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            # 不使用位置编码
            state_pos = None

        # 3. 扩展状态特征到批量大小
        state_feat = state_feat[None].expand(batch_size, -1, -1)

        return state_feat, state_pos, None

    def _decoder_explained(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
        """
        解码器方法解释

        这是模型的核心解码过程，它融合状态信息和图像信息，
        通过交叉注意力机制生成最终的特征表示。

        参数:
            f_state: [B, state_size, C] 状态特征
            pos_state: [B, state_size, 2] 状态位置编码
            f_img: [B, N, C] 图像特征
            pos_img: [B, N, 2] 图像位置编码
            f_pose: [B, 1, C] 位姿特征（可选）
            pos_pose: [B, 1, 2] 位姿位置编码（可选）

        返回:
            state_outputs: 状态特征的演化过程
            img_outputs: 图像特征的演化过程

        解码过程:
            1. 特征投影：将编码器特征投影到解码器空间
            2. 位姿融合：如果启用位姿头，融合位姿信息
            3. 交叉注意力：状态和图像特征相互交互
            4. 逐层细化：通过多层解码器逐步细化特征

        关键创新:
            - 双向交叉注意力：状态↔图像的双向信息流
            - 渐进式细化：每层都输出中间结果
            - 位姿集成：将位姿信息无缝集成到解码过程
            - 梯度检查点：支持大模型训练
        """
        # 1. 初始化输出列表，记录解码过程
        final_output = [(f_state, f_img)]  # 投影前的特征

        # 2. 确保状态特征维度正确
        assert f_state.shape[-1] == self.dec_embed_dim

        # 3. 将图像特征投影到解码器空间
        f_img = self.decoder_embed(f_img)

        # 4. 如果启用位姿头，融合位姿信息
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            # 将位姿特征添加到图像特征序列的开头
            f_img = torch.cat([f_pose, f_img], dim=1)
            pos_img = torch.cat([pos_pose, pos_img], dim=1)

        # 5. 记录投影后的特征
        final_output.append((f_state, f_img))

        # 6. 通过解码器块进行交叉注意力处理
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # 使用梯度检查点的情况
                # 状态块：状态特征关注图像特征
                f_state, _ = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],  # (f_state, f_img)
                    pos_state,
                    pos_img,
                    use_reentrant=not self.fixed_input_length,
                )
                # 图像块：图像特征关注状态特征
                f_img, _ = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],  # (f_img, f_state)
                    pos_img,
                    pos_state,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                # 正常前向传播
                # 状态特征通过交叉注意力关注图像特征
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)
                # 图像特征通过交叉注意力关注状态特征
                f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)

            # 记录每层的输出
            final_output.append((f_state, f_img))

        # 7. 移除重复的初始输出
        del final_output[1]  # 与final_output[0]重复

        # 8. 最终归一化
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )

        # 9. 返回状态和图像特征的演化过程
        return zip(*final_output)

    def forward_explained(self, views, ret_state=False):
        """
        前向传播方法解释

        这是模型的主要入口点，协调整个推理过程。它处理多视角输入，
        维护全局状态，并生成每个视角的预测结果。

        参数:
            views: List[Dict] 多视角输入数据
            ret_state: bool 是否返回状态信息

        返回:
            ARCroco3DStereoOutput: 包含预测结果的输出对象
            state_args: 状态信息（如果ret_state=True）

        处理流程:
            1. 编码所有视角：提取图像和射线特征
            2. 初始化状态：创建全局状态表示
            3. 逐视角处理：递归处理每个视角
            4. 状态更新：根据掩码更新全局状态
            5. 结果生成：通过头部网络生成最终预测

        关键特性:
            - 递归处理：支持任意长度的视频序列
            - 状态管理：维护跨视角的一致性
            - 掩码控制：灵活的更新和重置机制
            - 记忆系统：使用LocalMemory管理位姿信息
        """
        # 1. 编码所有视角的特征
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]  # 使用最后一层的特征

        # 2. 初始化全局状态
        state_feat, state_pos = self._init_state(feat[0], pos[0])

        # 3. 初始化位姿记忆系统
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()

        # 4. 记录所有状态参数（用于调试和分析）
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]

        # 5. 逐视角处理
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]

            # 5.1 处理位姿信息（如果启用位姿头）
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    # 第一帧使用初始位姿令牌
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    # 后续帧从记忆中查询位姿信息
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            # 5.2 递归解码过程
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )

            # 5.3 更新位姿记忆
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            # 5.4 生成预测结果
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),                           # 初始特征
                dec[self.dec_depth * 2 // 4][:, 1:].float(),  # 1/4深度特征
                dec[self.dec_depth * 3 // 4][:, 1:].float(),  # 3/4深度特征
                dec[self.dec_depth].float(),              # 最终特征
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)

            # 5.5 状态更新逻辑
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)

            # 确定更新掩码
            if update is not None:
                update_mask = img_mask & update  # 只有在img_mask和update都为True时才更新
            else:
                update_mask = img_mask

            update_mask = update_mask[:, None, None].float()

            # 更新全局状态
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
            # 更新局部记忆
            mem = new_mem * update_mask + mem * (1 - update_mask)

            # 处理重置掩码
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)

            # 记录状态参数
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )

        # 6. 返回结果
        if ret_state:
            return ress, views, all_state_args
        return ress, views


# ============================================================================
# 模型架构总结和设计思想
# ============================================================================

"""
CUT3R (ARCroco3DStereo) 模型架构总结

## 核心设计思想

1. **多模态融合**
   - RGB图像：提供视觉语义信息
   - 射线映射：提供几何约束信息
   - 位姿信息：提供空间关系约束

2. **状态管理机制**
   - 全局状态：维护跨视角的一致性信息
   - 局部记忆：存储和检索位姿相关的历史信息
   - 递归更新：支持长序列的在线处理

3. **交叉注意力架构**
   - 编码器：独立处理每种模态的信息
   - 解码器：通过交叉注意力融合多模态信息
   - 双向信息流：状态↔图像的相互增强

4. **多任务学习**
   - 3D重建：预测每个像素的3D坐标
   - 位姿估计：预测相机的6DOF位姿
   - 置信度估计：评估预测的可靠性
   - RGB重建：可选的颜色重建任务

## 关键技术创新

1. **LocalMemory机制**
   - 可学习的记忆库，存储历史位姿信息
   - 支持长期依赖关系的建模
   - 通过注意力机制进行读写操作

2. **状态递归处理**
   - 维护全局状态表示
   - 支持任意长度的视频序列
   - 灵活的更新和重置机制

3. **多尺度特征融合**
   - 使用不同深度的解码器特征
   - DPT头部进行多尺度融合
   - 提高预测的精度和鲁棒性

4. **自适应掩码机制**
   - 支持部分视角的处理
   - 灵活的状态更新控制
   - 处理遮挡和缺失数据

## 训练和推理特性

1. **梯度检查点**
   - 支持大模型的内存优化训练
   - 在训练时间和内存之间取得平衡

2. **混合精度训练**
   - 支持FP16训练加速
   - 保持数值稳定性

3. **灵活的配置系统**
   - 支持不同的头部网络类型
   - 可配置的输出模式
   - 适应不同的应用场景

4. **预训练权重兼容性**
   - 支持从CroCo预训练权重初始化
   - 灵活的权重加载机制
   - 向后兼容性保证

## 应用场景

1. **单目SLAM**
   - 实时位姿跟踪
   - 稠密地图构建
   - 回环检测

2. **多视角重建**
   - 静态场景重建
   - 动态对象跟踪
   - 3D内容创建

3. **增强现实**
   - 实时位姿估计
   - 虚拟对象放置
   - 遮挡处理

4. **机器人导航**
   - 视觉里程计
   - 障碍物检测
   - 路径规划

## 性能优化策略

1. **计算优化**
   - 使用RoPE位置编码减少计算量
   - 梯度检查点平衡内存和速度
   - 批量处理提高效率

2. **内存优化**
   - 固定大小的记忆库
   - 选择性的状态更新
   - 高效的注意力计算

3. **数值稳定性**
   - 层归一化防止梯度爆炸
   - 残差连接保持梯度流
   - 合适的初始化策略

## 扩展性和可定制性

1. **模块化设计**
   - 独立的编码器和解码器
   - 可插拔的头部网络
   - 灵活的配置系统

2. **多任务支持**
   - 可选的任务头部
   - 联合训练机制
   - 任务特定的损失函数

3. **数据适应性**
   - 支持不同分辨率的输入
   - 处理不同数量的视角
   - 适应不同的相机模型

这个架构代表了当前3D视觉领域的先进技术，结合了Transformer的强大表示能力、
多模态融合的优势，以及专门为3D重建和位姿估计设计的创新机制。
"""
