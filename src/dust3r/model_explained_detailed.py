# ============================================================================
# CUT3R模型详细解释版本 - 完整代码逐行注释
# 基于原始model.py文件，添加详细的中文注释和解释
# 作者：AI助手
# 目的：帮助理解CUT3R模型的完整架构和实现细节
# ============================================================================

import sys
import os

# 添加父目录到Python路径，用于导入croco模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 标准库导入
from collections import OrderedDict  # 有序字典，用于状态字典处理
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
from torch.utils.checkpoint import checkpoint  # 梯度检查点，节省内存
from copy import deepcopy  # 深拷贝
from functools import partial  # 偏函数，用于创建带默认参数的函数
from typing import Optional, Tuple, List, Any  # 类型提示
from dataclasses import dataclass  # 数据类装饰器

# HuggingFace Transformers相关导入
from transformers import PretrainedConfig  # 预训练配置基类
from transformers import PreTrainedModel  # 预训练模型基类
from transformers.modeling_outputs import BaseModelOutput  # 模型输出基类
from transformers.file_utils import ModelOutput  # 模型输出工具
import time  # 时间模块

# DUSt3R相关工具导入
from dust3r.utils.misc import (
    fill_default_args,      # 填充默认参数
    freeze_all_params,      # 冻结所有参数
    is_symmetrized,         # 检查是否对称化
    interleave,             # 交错排列
    transpose_to_landscape, # 转换为横向布局
)
from dust3r.heads import head_factory  # 头部网络工厂函数
from dust3r.utils.camera import PoseEncoder  # 位姿编码器
from dust3r.patch_embed import get_patch_embed  # 补丁嵌入获取函数

# 导入CroCo相关模块（需要先添加路径）
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa

# 导入自定义的Transformer块
from dust3r.blocks import (
    Block,              # 标准Transformer块
    DecoderBlock,       # 解码器块
    Mlp,                # 多层感知机
    Attention,          # 注意力机制
    CrossAttention,     # 交叉注意力
    DropPath,           # 随机深度
    CustomDecoderBlock, # 自定义解码器块
)  # noqa

# 定义无穷大常量
inf = float("inf")

# 导入加速库的日志记录器
from accelerate.logging import get_logger
printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    ARCroco3DStereo模型的自定义输出类
    
    继承自HuggingFace的ModelOutput，用于标准化模型输出格式。
    这个类定义了CUT3R模型的输出结构，包含预测结果和视图信息。
    
    属性:
        ress: 模型的预测结果列表，每个元素对应一个视图的预测
        views: 输入视图的信息列表，包含原始数据和元数据
    """
    ress: Optional[List[Any]] = None    # 预测结果列表
    views: Optional[List[Any]] = None   # 视图信息列表


def strip_module(state_dict):
    """
    从状态字典的键中移除'module.'前缀
    
    在使用DataParallel或DistributedDataParallel训练时，模型参数会被包装
    在'module.'前缀下。这个函数用于清理这些前缀，使状态字典可以正确加载。
    
    参数:
        state_dict (dict): 原始状态字典，可能包含'module.'前缀
        
    返回:
        OrderedDict: 清理后的状态字典，移除了'module.'前缀
        
    示例:
        原始键: 'module.encoder.weight' -> 清理后: 'encoder.weight'
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 检查键是否以'module.'开头，如果是则移除前缀
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    """
    从检查点文件加载CUT3R模型
    
    这个函数处理模型加载的完整流程，包括：
    1. 加载检查点文件
    2. 解析模型配置参数
    3. 实例化模型
    4. 加载预训练权重
    
    参数:
        model_path (str): 模型检查点文件路径
        device (str): 目标设备 ('cpu' 或 'cuda')
        verbose (bool): 是否打印详细信息
        
    返回:
        ARCroco3DStereo: 加载完成的模型实例
    """
    if verbose:
        print("... loading model from", model_path)
    
    # 加载检查点到CPU（避免GPU内存问题）
    ckpt = torch.load(model_path, map_location="cpu")
    
    # 获取模型配置参数字符串
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR只用于宽高比不一致的情况
    
    # 处理landscape_only参数的兼容性
    if "landscape_only" not in args:
        # 如果配置中没有landscape_only参数，添加默认值False
        args = args[:-2] + ", landscape_only=False))"
    else:
        # 如果有该参数，确保设置为False
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    
    # 确保landscape_only参数正确设置
    assert "landscape_only=False" in args
    
    if verbose:
        print(f"instantiating : {args}")
    
    # 使用eval动态实例化模型（从配置字符串）
    net = eval(args)
    
    # 加载预训练权重，strict=False允许部分匹配
    s = net.load_state_dict(ckpt["model"], strict=False)
    
    if verbose:
        print(s)  # 打印加载状态信息
    
    # 将模型移动到指定设备
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    """
    ARCroco3DStereo模型的配置类
    
    继承自HuggingFace的PretrainedConfig，定义了CUT3R模型的所有配置参数。
    这个类包含了模型架构、训练设置、输出模式等所有可配置的选项。
    """
    model_type = "arcroco_3d_stereo"  # 模型类型标识符

    def __init__(
        self,
        # === 输出和头部配置 ===
        output_mode="pts3d",                    # 输出模式：'pts3d', 'pts3d+pose'等
        head_type="linear",                     # 头部类型：'linear' 或 'dpt'
        depth_mode=("exp", -float("inf"), float("inf")),  # 深度模式和范围
        conf_mode=("exp", 1, float("inf")),     # 置信度模式和范围
        pose_mode=("exp", -float("inf"), float("inf")),   # 位姿模式和范围
        
        # === 训练配置 ===
        freeze="none",                          # 冻结策略：'none', 'encoder'等
        landscape_only=True,                    # 是否只处理横向图像
        
        # === 补丁嵌入配置 ===
        patch_embed_cls="PatchEmbedDust3R",     # 补丁嵌入类名
        
        # === 射线编码器配置 ===
        ray_enc_depth=2,                        # 射线编码器深度
        
        # === 状态管理配置 ===
        state_size=324,                         # 状态向量大小
        local_mem_size=256,                     # 局部记忆大小
        state_pe="2d",                          # 状态位置编码类型
        state_dec_num_heads=16,                 # 状态解码器注意力头数
        
        # === 任务头部开关 ===
        depth_head=False,                       # 是否启用深度头
        rgb_head=False,                         # 是否启用RGB头
        pose_conf_head=False,                   # 是否启用位姿置信度头
        pose_head=False,                        # 是否启用位姿头
        
        **croco_kwargs,                         # CroCo的其他参数
    ):
        """
        初始化配置参数
        
        参数说明:
            output_mode: 定义模型输出什么信息
            head_type: 选择使用线性头还是DPT头（DPT质量更高但计算量大）
            depth_mode/conf_mode/pose_mode: 定义各输出的激活函数和范围
            freeze: 训练时冻结哪些部分
            state_size: 全局状态的维度大小
            local_mem_size: LocalMemory的大小
            各种head开关: 控制启用哪些输出任务
        """
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
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    """
    局部记忆模块 - CUT3R的核心创新之一

    这个模块实现了一个可学习的记忆库，用于存储和检索历史位姿信息。
    它是CUT3R实现连续3D感知的关键组件，支持长期依赖关系的建模。

    核心思想:
    1. 维护一个可学习的记忆库，存储历史位姿相关信息
    2. 通过注意力机制进行读写操作
    3. 支持动态更新和查询，实现长序列的连续处理

    架构设计:
    - 记忆库: 可学习的参数矩阵，存储历史信息
    - 写入块: 用于更新记忆库的Transformer解码器块
    - 读取块: 用于从记忆库检索信息的Transformer解码器块
    """

    def __init__(
        self,
        size,                           # 记忆库大小（记忆槽数量）
        k_dim,                          # 键的维度
        v_dim,                          # 值的维度
        num_heads,                      # 注意力头数
        depth=2,                        # Transformer块深度
        mlp_ratio=4.0,                  # MLP扩展比例
        qkv_bias=False,                 # 是否使用QKV偏置
        drop=0.0,                       # Dropout概率
        attn_drop=0.0,                  # 注意力Dropout概率
        drop_path=0.0,                  # DropPath概率
        act_layer=nn.GELU,              # 激活函数
        norm_layer=nn.LayerNorm,        # 归一化层
        norm_mem=True,                  # 是否对记忆进行归一化
        rope=None,                      # 旋转位置编码
    ) -> None:
        """
        初始化LocalMemory模块

        参数说明:
            size: 记忆库的容量，即可以存储多少个记忆槽
            k_dim: 查询键的维度，通常是编码器的输出维度
            v_dim: 记忆值的维度，通常是解码器的输入维度
            num_heads: 多头注意力的头数
            depth: 读写Transformer块的层数
            其他参数: 标准Transformer参数
        """
        super().__init__()
        self.v_dim = v_dim

        # === 查询投影层 ===
        # 将输入的键维度投影到值维度
        self.proj_q = nn.Linear(k_dim, v_dim)


        # === 掩码token ===
        # 用于读取操作时的占位符token，可学习参数
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )


        # === 核心记忆库 ===
        # 可学习的记忆参数矩阵，存储历史信息
        # 维度: [1, size, 2*v_dim] - 2*v_dim是因为存储键值对
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )

        # === 写入Transformer块 ===
        # 用于更新记忆库的Transformer解码器块列表
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,              # 输入维度（键值对）
                    num_heads,              # 注意力头数
                    mlp_ratio=mlp_ratio,    # MLP扩展比例
                    qkv_bias=qkv_bias,      # QKV偏置
                    norm_layer=norm_layer,  # 归一化层
                    attn_drop=attn_drop,    # 注意力dropout
                    drop=drop,              # 普通dropout
                    drop_path=drop_path,    # 随机深度
                    act_layer=act_layer,    # 激活函数
                    norm_mem=norm_mem,      # 记忆归一化
                    rope=rope,              # 位置编码
                )
                for _ in range(depth)  # 创建depth个写入块
            ]
        )

        # === 读取Transformer块 ===
        # 用于从记忆库检索信息的Transformer解码器块列表
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,              # 输入维度（键值对）
                    num_heads,              # 注意力头数
                    mlp_ratio=mlp_ratio,    # MLP扩展比例
                    qkv_bias=qkv_bias,      # QKV偏置
                    norm_layer=norm_layer,  # 归一化层
                    attn_drop=attn_drop,    # 注意力dropout
                    drop=drop,              # 普通dropout
                    drop_path=drop_path,    # 随机深度
                    act_layer=act_layer,    # 激活函数
                    norm_mem=norm_mem,      # 记忆归一化
                    rope=rope,              # 位置编码
                )
                for _ in range(depth)  # 创建depth个读取块
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        更新记忆库 - 写入操作

        这个方法将新的特征信息写入记忆库。通过Transformer解码器块
        处理新信息和现有记忆，实现记忆的动态更新。

        参数:
            mem: 当前记忆库状态 [B, size, 2*C]
            feat_k: 新的键特征 [B, 1, C] - 通常是图像级特征
            feat_v: 新的值特征 [B, 1, C] - 通常是位姿相关特征

        返回:
            更新后的记忆库 [B, size, 2*C]

        工作流程:
        1. 将键特征投影到值维度
        2. 拼接键值特征形成完整的新信息
        3. 通过写入Transformer块处理记忆更新
        4. 返回更新后的记忆库
        """
        # 步骤1: 投影键特征到值维度
        feat_k = self.proj_q(feat_k)  # [B, 1, C] -> [B, 1, v_dim]

        # 步骤2: 拼接键值特征
        feat = torch.cat([feat_k, feat_v], dim=-1)  # [B, 1, 2*v_dim]

        # 步骤3: 通过写入块更新记忆
        for blk in self.write_blocks:
            # 记忆作为查询，新特征作为键值
            # 这样记忆可以选择性地吸收新信息
            mem, _ = blk(mem, feat, None, None)

        return mem

    def inquire(self, query, mem):
        """
        从记忆库检索信息 - 读取操作

        这个方法根据查询从记忆库中检索相关信息。通过注意力机制
        找到与当前查询最相关的历史信息。

        参数:
            query: 查询特征 [B, 1, k_dim] - 通常是当前帧的图像特征
            mem: 记忆库状态 [B, size, 2*v_dim]

        返回:
            检索到的信息 [B, 1, v_dim] - 与查询相关的历史信息

        工作流程:
        1. 将查询投影到值维度
        2. 添加掩码token作为输出占位符
        3. 通过读取Transformer块处理查询和记忆
        4. 提取并返回检索结果
        """
        # 步骤1: 投影查询到值维度
        x = self.proj_q(query)  # [B, 1, k_dim] -> [B, 1, v_dim]

        # 步骤2: 添加掩码token作为输出占位符
        # 掩码token将通过注意力机制聚合记忆信息
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        # 结果: [B, 1, 2*v_dim] = [查询, 掩码token]

        # 步骤3: 通过读取块处理查询
        for blk in self.read_blocks:
            # x作为查询，mem作为键值
            # 掩码token会通过注意力聚合相关的记忆信息
            x, _ = blk(x, mem, None, None)

        # 步骤4: 提取检索结果
        # 只返回掩码token部分，它现在包含了检索到的信息
        return x[..., -self.v_dim :]  # [B, 1, v_dim]


class ARCroco3DStereo(CroCoNet):
    """
    ARCroco3DStereo - CUT3R的主模型类

    这是CUT3R模型的核心实现，继承自CroCoNet（CroCo的基础网络）。
    该模型实现了连续3D感知的完整架构，包括：

    核心特性:
    1. 双路编码器: 分别处理RGB图像和射线映射
    2. 状态管理: 维护跨帧的全局状态表示
    3. 记忆机制: 使用LocalMemory存储历史位姿信息
    4. 双解码器: 状态解码器和图像解码器交替处理
    5. 多任务输出: 3D重建、位姿估计、RGB重建等

    架构继承关系:
    ARCroco3DStereo -> CroCoNet -> PreTrainedModel
    """

    # HuggingFace模型配置
    config_class = ARCroco3DStereoConfig      # 配置类
    base_model_prefix = "arcroco3dstereo"     # 模型前缀
    supports_gradient_checkpointing = True    # 支持梯度检查点

    def __init__(self, config: ARCroco3DStereoConfig):
        """
        初始化ARCroco3DStereo模型

        这个初始化过程包括：
        1. 继承CroCo的基础架构
        2. 添加射线映射编码器
        3. 设置状态管理组件
        4. 配置位姿检索记忆
        5. 设置双解码器架构
        6. 初始化输出头部

        参数:
            config: ARCroco3DStereoConfig配置对象
        """
        # === 基础设置 ===
        self.gradient_checkpointing = False     # 梯度检查点开关
        self.fixed_input_length = True          # 固定输入长度

        # === 配置处理 ===
        # 填充CroCo配置的默认参数
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs

        # === 初始化CroCo基础架构 ===
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)  # 调用CroCoNet的初始化

        # === 射线映射编码器 ===
        # 专门用于处理射线映射的Transformer编码器块
        # 射线映射包含6个通道：射线方向(3) + 射线原点(3)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,         # 编码器嵌入维度
                    16,                         # 注意力头数（固定为16）
                    4,                          # MLP比例
                    qkv_bias=True,              # 使用QKV偏置
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 层归一化
                    rope=self.rope,             # 旋转位置编码
                )
                for _ in range(config.ray_enc_depth)  # 创建指定深度的块
            ]
        )

        # 射线映射编码器的归一化层
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)

        # === 解码器配置 ===
        self.dec_num_heads = self.croco_args["dec_num_heads"]

        # === 位姿头部设置 ===
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            # 位姿token: 用于位姿估计的可学习token
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )

            # 位姿检索器: LocalMemory实例，用于存储和检索历史位姿信息
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,        # 记忆库大小
                k_dim=self.enc_embed_dim,          # 键维度（编码器输出）
                v_dim=self.dec_embed_dim,          # 值维度（解码器输入）
                num_heads=self.dec_num_heads,      # 注意力头数
                mlp_ratio=4,                       # MLP扩展比例
                qkv_bias=True,                     # QKV偏置
                attn_drop=0.0,                     # 注意力dropout
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 归一化层
                rope=None,                         # 不使用位置编码
            )

        # === 状态管理组件 ===
        # 注册token: 用于初始化全局状态的可学习嵌入
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe

        # === 掩码token ===
        # 用于处理缺失图像的掩码token
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )

        # 用于处理缺失射线映射的掩码token
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )

        # === 状态解码器设置 ===
        # 设置专门处理状态的解码器
        self._set_state_decoder(
            self.enc_embed_dim,                 # 编码器维度
            self.dec_embed_dim,                 # 解码器维度
            config.state_dec_num_heads,         # 状态解码器注意力头数
            self.dec_depth,                     # 解码器深度
            self.croco_args.get("mlp_ratio", None),      # MLP比例
            self.croco_args.get("norm_layer", None),     # 归一化层
            self.croco_args.get("norm_im2_in_dec", None), # 解码器中的图像归一化
        )

        # === 下游任务头部设置 ===
        # 根据配置设置各种输出头部（3D点云、位姿、RGB等）
        self.set_downstream_head(
            config.output_mode,         # 输出模式
            config.head_type,           # 头部类型
            config.landscape_only,      # 是否只处理横向图像
            config.depth_mode,          # 深度模式
            config.conf_mode,           # 置信度模式
            config.pose_mode,           # 位姿模式
            config.depth_head,          # 深度头开关
            config.rgb_head,            # RGB头开关
            config.pose_conf_head,      # 位姿置信度头开关
            config.pose_head,           # 位姿头开关
            **self.croco_args,          # 其他CroCo参数
        )

        # === 参数冻结设置 ===
        # 根据配置冻结指定的模型部分
        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        """
        从预训练模型加载CUT3R模型

        这个类方法支持两种加载方式：
        1. 从本地文件路径加载检查点
        2. 从HuggingFace Hub加载模型

        参数:
            pretrained_model_name_or_path: 模型路径或HuggingFace模型名
            **kw: 其他关键字参数

        返回:
            加载完成的ARCroco3DStereo模型实例
        """
        if os.path.isfile(pretrained_model_name_or_path):
            # 情况1: 本地文件路径 - 使用自定义加载函数
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            # 情况2: HuggingFace模型名 - 使用父类方法
            try:
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

        CUT3R使用双路补丁嵌入：
        1. 标准图像补丁嵌入（3通道RGB）
        2. 射线映射补丁嵌入（6通道：方向+原点）

        参数:
            img_size: 输入图像尺寸
            patch_size: 补丁大小
            enc_embed_dim: 编码器嵌入维度
        """
        # === 标准图像补丁嵌入 ===
        # 将RGB图像分割成补丁并嵌入到特征空间
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls,   # 补丁嵌入类名
            img_size,               # 图像尺寸
            patch_size,             # 补丁尺寸
            enc_embed_dim,          # 嵌入维度
            in_chans=3              # 输入通道数（RGB）
        )

        # === 射线映射补丁嵌入 ===
        # 将射线映射分割成补丁并嵌入到特征空间
        # 射线映射包含6个通道：射线方向(3) + 射线原点(3)
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls,   # 补丁嵌入类名
            img_size,               # 图像尺寸
            patch_size,             # 补丁尺寸
            enc_embed_dim,          # 嵌入维度
            in_chans=6              # 输入通道数（射线映射）
        )

    def _set_decoder(
        self,
        enc_embed_dim,      # 编码器嵌入维度
        dec_embed_dim,      # 解码器嵌入维度
        dec_num_heads,      # 解码器注意力头数
        dec_depth,          # 解码器深度
        mlp_ratio,          # MLP扩展比例
        norm_layer,         # 归一化层
        norm_im2_in_dec,    # 解码器中的图像归一化
    ):
        """
        设置标准解码器

        这个方法设置CroCo的标准解码器，用于处理图像特征。
        在CUT3R中，这个解码器与状态解码器配合工作。

        参数说明:
            enc_embed_dim: 编码器输出的特征维度
            dec_embed_dim: 解码器内部的特征维度
            dec_num_heads: 多头注意力的头数
            dec_depth: 解码器的层数
            mlp_ratio: MLP层的扩展比例
            norm_layer: 使用的归一化层类型
            norm_im2_in_dec: 是否在解码器中对第二个图像进行归一化
        """
        # 保存解码器配置
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim

        # === 编码器到解码器的投影层 ===
        # 将编码器的输出特征投影到解码器的输入维度
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)

        # === 解码器Transformer块列表 ===
        # 创建指定深度的解码器块，每个块包含交叉注意力和自注意力
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,          # 解码器维度
                    dec_num_heads,          # 注意力头数
                    mlp_ratio=mlp_ratio,    # MLP扩展比例
                    qkv_bias=True,          # 使用QKV偏置
                    norm_layer=norm_layer,  # 归一化层
                    norm_mem=norm_im2_in_dec,  # 记忆归一化
                    rope=self.rope,         # 旋转位置编码
                )
                for _ in range(dec_depth)  # 创建dec_depth个解码器块
            ]
        )

        # === 解码器输出归一化层 ===
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,          # 编码器嵌入维度
        dec_embed_dim,          # 解码器嵌入维度
        state_dec_num_heads,    # 状态解码器注意力头数
        dec_depth,              # 解码器深度
        mlp_ratio,              # MLP扩展比例
        norm_layer,             # 归一化层
        norm_im2_in_dec,        # 解码器中的图像归一化
    ):
        """
        设置状态解码器 - CUT3R的核心创新

        状态解码器是CUT3R实现连续3D感知的关键组件。它与标准解码器
        配合工作，实现状态和图像特征的双向交互。

        架构设计:
        - 状态解码器: 处理全局状态的更新
        - 图像解码器: 处理图像特征的细化
        - 双向交互: 状态 ↔ 图像的相互增强

        参数说明:
            enc_embed_dim: 编码器输出维度
            dec_embed_dim: 解码器内部维度
            state_dec_num_heads: 状态解码器的注意力头数
            其他参数: 与标准解码器相同
        """
        # === 状态到解码器的投影层 ===
        # 将状态特征投影到解码器维度
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)

        # === 状态解码器Transformer块列表 ===
        # 专门处理状态更新的解码器块
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,              # 解码器维度
                    state_dec_num_heads,        # 状态解码器注意力头数
                    mlp_ratio=mlp_ratio,        # MLP扩展比例
                    qkv_bias=True,              # 使用QKV偏置
                    norm_layer=norm_layer,      # 归一化层
                    norm_mem=norm_im2_in_dec,   # 记忆归一化
                    rope=self.rope,             # 旋转位置编码
                )
                for _ in range(dec_depth)      # 与图像解码器相同深度
            ]
        )

        # === 状态解码器输出归一化层 ===
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        """
        加载模型状态字典的自定义方法

        这个方法处理CUT3R模型加载预训练权重时的兼容性问题。
        由于模型架构的演进，不同版本的检查点可能有不同的键名，
        这个方法提供了多层次的回退机制来确保成功加载。

        处理的兼容性问题:
        1. 移除DataParallel的'module.'前缀
        2. 处理解码器块名称的变化 (dec_blocks -> dec_blocks_state)
        3. 处理尺寸不匹配的参数
        4. 跳过不存在的参数

        参数:
            ckpt (dict): 检查点状态字典
            **kw: 传递给父类load_state_dict的其他参数

        返回:
            加载结果信息
        """
        # === 步骤1: 处理DataParallel前缀 ===
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)

        new_ckpt = dict(ckpt)

        # === 步骤2: 处理解码器块名称兼容性 ===
        # 旧版本使用dec_blocks，新版本使用dec_blocks_state
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    # 复制dec_blocks的权重到dec_blocks_state
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value

        # === 步骤3: 尝试直接加载 ===
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            # === 步骤4: 第一层回退 - 跳过解码器相关参数 ===
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                # === 步骤5: 最后回退 - 只加载匹配的参数 ===
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        # 检查参数尺寸是否匹配
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        """
        设置模型参数冻结策略

        这个方法用于在微调或下游任务中冻结模型的特定部分。
        通过冻结预训练的参数，可以防止过拟合并加速训练。

        冻结策略:
        - "none": 不冻结任何参数，全模型训练
        - "mask": 只冻结掩码token（如果存在）
        - "encoder": 冻结所有编码器相关参数
        - "encoder_and_head": 冻结编码器和下游头部
        - "encoder_and_decoder": 冻结编码器和解码器
        - "decoder": 只冻结解码器相关参数

        参数:
            freeze (str): 冻结策略名称
        """
        self.freeze = freeze

        # === 定义不同冻结策略对应的参数组 ===
        to_be_frozen = {
            "none": [],  # 不冻结任何参数

            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],

            # 冻结所有编码器组件
            "encoder": [
                self.patch_embed,           # 图像补丁嵌入
                self.patch_embed_ray_map,   # 射线映射补丁嵌入
                self.masked_img_token,      # 图像掩码token
                self.masked_ray_map_token,  # 射线掩码token
                self.enc_blocks,            # 图像编码器块
                self.enc_blocks_ray_map,    # 射线编码器块
                self.enc_norm,              # 图像编码器归一化
                self.enc_norm_ray_map,      # 射线编码器归一化
            ],

            # 冻结编码器和下游头部
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,       # 下游任务头部
            ],

            # 冻结编码器和解码器（几乎整个模型）
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,            # 主解码器块
                self.dec_blocks_state,      # 状态解码器块
                self.pose_retriever,        # 位姿检索器
                self.pose_token,            # 位姿token
                self.register_tokens,       # 状态注册token
                self.decoder_embed_state,   # 状态解码器嵌入
                self.decoder_embed,         # 主解码器嵌入
                self.dec_norm,              # 主解码器归一化
                self.dec_norm_state,        # 状态解码器归一化
            ],

            # 只冻结解码器相关参数
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }

        # === 执行参数冻结 ===
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """
        设置预测头部的占位方法

        这是一个占位方法，CUT3R不使用传统的预测头部，
        而是使用更复杂的下游任务头部架构。
        实际的头部设置在set_downstream_head方法中完成。
        """
        return

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
        patch_size,
        img_size,
        **kw,
    ):
        """
        设置下游任务头部

        这个方法配置CUT3R的输出头部，支持多种任务和输出模式。
        头部负责将解码器的特征转换为最终的预测结果。

        支持的任务:
        - 3D点云重建 (pts3d)
        - 深度估计 (depth)
        - RGB重建 (rgb)
        - 位姿估计 (pose)
        - 置信度估计 (confidence)

        参数:
            output_mode (str): 输出模式，如"pts3d", "pts3d+pose"等
            head_type (str): 头部类型，"linear"或"dpt"
            landscape_only (bool): 是否只处理横向图像
            depth_mode (tuple): 深度输出的激活函数和范围
            conf_mode (tuple): 置信度输出的激活函数和范围
            pose_mode (tuple): 位姿输出的激活函数和范围
            depth_head (bool): 是否启用深度头部
            rgb_head (bool): 是否启用RGB头部
            pose_conf_head (bool): 是否启用位姿置信度头部
            pose_head (bool): 是否启用位姿头部
            patch_size (int): 补丁大小
            img_size (tuple): 图像尺寸
        """
        # === 验证图像尺寸和补丁大小的兼容性 ===
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"

        # === 保存配置参数 ===
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode

        # === 创建下游任务头部 ===
        # 使用工厂函数创建适当的头部架构
        self.downstream_head = head_factory(
            head_type,                      # 头部类型（linear/dpt）
            output_mode,                    # 输出模式
            self,                           # 模型实例
            has_conf=bool(conf_mode),       # 是否有置信度输出
            has_depth=bool(depth_head),     # 是否有深度输出
            has_rgb=bool(rgb_head),         # 是否有RGB输出
            has_pose_conf=bool(pose_conf_head),  # 是否有位姿置信度
            has_pose=bool(pose_head),       # 是否有位姿输出
        )

        # === 应用横向图像转换 ===
        # 如果启用landscape_only，会自动处理图像方向
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        """
        编码RGB图像

        这个方法将输入的RGB图像编码为特征表示。它是CUT3R双路
        编码器的第一路，专门处理视觉语义信息。

        处理流程:
        1. 补丁嵌入: 将图像分割成补丁并嵌入
        2. Transformer编码: 通过24层Transformer提取特征
        3. 归一化: 对输出特征进行归一化

        参数:
            image: 输入RGB图像 [B, 3, H, W]
            true_shape: 图像的真实尺寸 [B, 2]

        返回:
            tuple: (特征列表, 位置编码, None)
            - 特征列表: [[B, N, D]] 其中N是补丁数，D是特征维度
            - 位置编码: [B, N, 2] 每个补丁的2D位置
        """
        # === 步骤1: 补丁嵌入 ===
        # 将图像分割成补丁并嵌入到特征空间
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x: [B, N, enc_embed_dim] 其中N = (H/patch_size) * (W/patch_size)
        # pos: [B, N, 2] 每个补丁的归一化2D坐标

        # === 位置编码检查 ===
        # CUT3R使用RoPE位置编码，不需要额外的位置嵌入
        assert self.enc_pos_embed is None

        # === 步骤2: Transformer编码 ===
        # 通过24层Transformer编码器提取深层特征
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                # 训练时使用梯度检查点节省内存
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                # 推理时直接前向传播
                x = blk(x, pos)  # 自注意力 + MLP + 残差连接

        # === 步骤3: 归一化 ===
        x = self.enc_norm(x)

        # 返回格式: ([特征], 位置, None)
        # 列表格式是为了与多尺度特征兼容
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        """
        编码射线映射

        这个方法将射线映射编码为特征表示。它是CUT3R双路编码器的
        第二路，专门处理几何约束信息。

        射线映射包含:
        - 射线方向 (3通道): 每个像素对应的3D射线方向
        - 射线原点 (3通道): 每个像素对应的3D射线起点

        处理流程:
        1. 补丁嵌入: 将射线映射分割成补丁并嵌入
        2. 专用编码: 通过2层专用Transformer提取几何特征
        3. 归一化: 对输出特征进行归一化

        参数:
            ray_map: 输入射线映射 [B, 6, H, W]
            true_shape: 图像的真实尺寸 [B, 2]

        返回:
            tuple: (特征列表, 位置编码, None)
        """
        # === 步骤1: 射线映射补丁嵌入 ===
        # 将6通道射线映射分割成补丁并嵌入
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        # x: [B, N, enc_embed_dim] 射线映射的补丁特征
        # pos: [B, N, 2] 每个补丁的2D位置（与图像相同）

        # === 位置编码检查 ===
        assert self.enc_pos_embed is None

        # === 步骤2: 射线映射专用编码 ===
        # 使用专门的2层Transformer编码器处理几何信息
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                # 训练时使用梯度检查点
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                # 推理时直接前向传播
                x = blk(x, pos)  # 几何特征的自注意力处理

        # === 步骤3: 射线映射归一化 ===
        x = self.enc_norm_ray_map(x)

        # 返回格式与图像编码器相同
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        """
        编码全局状态

        这个方法初始化CUT3R的全局状态表示。全局状态是连续3D感知
        的核心，它维护跨帧的场景信息和几何约束。

        状态设计思想:
        1. 可学习的状态token: 不依赖于具体输入的通用表示
        2. 2D位置编码: 为状态token提供空间结构
        3. 动态初始化: 根据第一帧图像特征调整状态

        参数:
            image_tokens: 第一帧图像的编码特征 [B, N, D]
            image_pos: 第一帧图像的位置编码 [B, N, 2]

        返回:
            tuple: (状态特征, 状态位置编码, None)
        """
        batch_size = image_tokens.shape[0]

        # === 步骤1: 获取可学习的状态token ===
        # 从嵌入层获取state_size个可学习的状态token
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )
        # state_feat: [state_size, enc_embed_dim]

        # === 步骤2: 生成状态位置编码 ===
        if self.state_pe == "1d":
            # 1D位置编码: 简单的线性排列
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # [B, state_size, 2]

        elif self.state_pe == "2d":
            # 2D位置编码: 将状态token排列成2D网格
            width = int(self.state_size**0.5)  # 计算网格宽度
            width = width + 1 if width % 2 == 1 else width  # 确保偶数宽度

            # 生成2D网格坐标
            state_pos = (
                torch.tensor(
                    [
                        [i % width, i // width]  # (x, y) 坐标
                        for i in range(self.state_size)
                    ],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # [B, state_size, 2]

        # === 步骤3: 扩展到批次维度 ===
        # 将状态特征扩展到批次维度
        state_feat = state_feat[None].expand(batch_size, -1, -1).contiguous()
        # state_feat: [B, state_size, enc_embed_dim]

        # 状态位置已经在上面扩展过了
        # state_pos: [B, state_size, 2]

        return state_feat, state_pos, None

    def _init_state(self, image_tokens, image_pos):
        """
        初始化全局状态

        这个方法是状态管理的入口点，它：
        1. 调用_encode_state生成初始状态
        2. 将状态投影到解码器维度
        3. 为后续的递归处理做准备

        参数:
            image_tokens: 第一帧图像特征 [B, N, D]
            image_pos: 第一帧位置编码 [B, N, 2]

        返回:
            tuple: (投影后的状态特征, 状态位置编码)
        """
        # === 步骤1: 编码状态 ===
        # 生成初始的状态表示
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)

        # === 步骤2: 投影到解码器维度 ===
        # 将状态特征从编码器维度投影到解码器维度
        state_feat = self.decoder_embed_state(state_feat)
        # state_feat: [B, state_size, dec_embed_dim]

        return state_feat, state_pos

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        """
        编码多视图输入 - CUT3R双路编码器的核心方法

        这个方法是CUT3R特征提取的核心，它实现了：
        1. 双路编码器的并行处理（图像 + 射线映射）
        2. 基于掩码的模态选择和融合
        3. 统一的特征输出格式

        处理流程:
        1. 提取和处理掩码信息
        2. 分别编码图像和射线映射
        3. 基于掩码融合两种特征
        4. 处理缺失模态的情况

        参数:
            views: 视图列表，每个视图包含img, ray_map, img_mask, ray_mask等
            img_mask: 图像掩码 [num_views, batch_size] (可选)
            ray_mask: 射线掩码 [num_views, batch_size] (可选)

        返回:
            tuple: (形状信息, 特征列表, 位置编码)
            - 形状信息: 每个视图的图像尺寸
            - 特征列表: 融合后的特征表示
            - 位置编码: 对应的位置信息
        """
        # === 步骤1: 基础信息提取 ===
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]

        # === 步骤2: 掩码处理 ===
        given = True
        if img_mask is None and ray_mask is None:
            given = False

        if not given:
            # 从视图中提取掩码信息
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)

        # === 步骤3: 数据准备 ===
        # 堆叠所有视图的图像
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)

        # 堆叠所有视图的射线映射
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)

        # 处理图像形状信息
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                # 如果没有真实形状，使用图像的实际尺寸
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)

        # === 步骤4: 数据重塑 ===
        # 将视图维度和批次维度合并，便于批量处理
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)

        # 展平掩码
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)

        # === 步骤5: 图像编码 ===
        # 只编码被掩码选中的图像
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]

        if selected_imgs.size(0) > 0:
            # 调用图像编码器
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            # 如果没有图像需要编码，抛出异常
            raise NotImplementedError

        # === 步骤6: 初始化统一输出张量 ===
        # 为所有视图创建统一的特征张量
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )

        # === 步骤7: 图像特征填充 ===
        for i in range(len(img_out)):
            # 将图像特征填充到对应位置
            full_out[i][img_masks_flat] += img_out[i]
            # 为没有图像的位置填充掩码token
            full_out[i][~img_masks_flat] += self.masked_img_token

        # 填充位置编码
        full_pos[img_masks_flat] += img_pos

        # === 步骤8: 射线映射编码 ===
        # 调整射线映射的维度顺序：(N, H, W, C) -> (N, C, H, W)
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]

        if selected_ray_maps.size(0) > 0:
            # 调用射线映射编码器
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )

            # 确保输出层数一致
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"

            # === 步骤9: 射线特征融合 ===
            for i in range(len(ray_out)):
                # 将射线特征添加到对应位置（与图像特征相加融合）
                full_out[i][ray_masks_flat] += ray_out[i]
                # 为没有射线的位置填充掩码token
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token

            # 位置编码处理：只在没有图像掩码的位置使用射线位置编码
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            # === 步骤10: 处理没有射线映射的情况 ===
            # 创建零射线映射用于保持代码一致性
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True

            # 编码零射线映射
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])

            # 填充零特征（实际上不起作用，因为乘以0.0）
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0

        # === 步骤11: 输出格式化 ===
        # 将合并的张量重新分割为视图格式
        return (
            shapes.chunk(len(views), dim=0),                    # 形状信息
            [out.chunk(len(views), dim=0) for out in full_out], # 特征列表
            full_pos.chunk(len(views), dim=0),                  # 位置编码
        )

    def _recurrent_rollout(
        self,
        state_feat,         # 当前状态特征
        state_pos,          # 状态位置编码
        current_feat,       # 当前帧特征
        current_pos,        # 当前帧位置编码
        pose_feat,          # 位姿特征（可选）
        pose_pos,           # 位姿位置编码（可选）
        init_state_feat,    # 初始状态特征
        img_mask=None,      # 图像掩码
        reset_mask=None,    # 重置掩码
        update=None,        # 更新掩码
    ):
        """
        递归状态更新

        这是CUT3R连续处理的核心方法。它实现了状态的递归更新，
        使模型能够处理任意长度的视频序列。

        处理流程:
        1. 调用双解码器处理当前帧和状态
        2. 提取更新后的状态特征
        3. 返回新状态和解码器输出

        参数:
            state_feat: 当前全局状态 [B, state_size, dec_embed_dim]
            state_pos: 状态位置编码 [B, state_size, 2]
            current_feat: 当前帧特征 [B, N, enc_embed_dim]
            current_pos: 当前帧位置 [B, N, 2]
            pose_feat: 位姿特征 [B, 1, dec_embed_dim] (可选)
            pose_pos: 位姿位置 [B, 1, 2] (可选)
            init_state_feat: 初始状态特征 [B, state_size, dec_embed_dim]
            其他参数: 各种掩码，用于控制更新行为

        返回:
            tuple: (新状态特征, 解码器输出列表)
        """
        # === 调用双解码器 ===
        # 这是CUT3R的核心：状态解码器和图像解码器的交替处理
        new_state_feat, dec = self._decoder(
            state_feat,     # 全局状态作为查询
            state_pos,      # 状态位置编码
            current_feat,   # 当前帧特征作为键值
            current_pos,    # 当前帧位置编码
            pose_feat,      # 位姿特征（如果启用）
            pose_pos,       # 位姿位置编码
        )

        # === 提取最终状态 ===
        # 取最后一层解码器的状态输出作为新的全局状态
        new_state_feat = new_state_feat[-1]

        return new_state_feat, dec

    def _get_img_level_feat(self, feat):
        """
        获取图像级特征

        这个方法将补丁级特征聚合为图像级特征，用于位姿估计
        和记忆更新。通过平均池化实现特征聚合。

        参数:
            feat: 补丁级特征 [B, N, D]

        返回:
            图像级特征 [B, 1, D]
        """
        # 对所有补丁特征进行平均池化
        # 这样得到一个代表整个图像的全局特征向量
        return torch.mean(feat, dim=1, keepdim=True)

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
        """
        双解码器核心实现 - CUT3R的关键创新

        这是CUT3R最重要的方法之一，实现了状态解码器和图像解码器
        的交替处理。这种设计使得全局状态和局部图像特征能够相互
        增强，实现更好的3D理解。

        架构设计:
        1. 状态解码器: 更新全局状态，维护场景的整体理解
        2. 图像解码器: 细化图像特征，提取局部细节信息
        3. 交替处理: 两个解码器在每一层都进行信息交换
        4. 位姿集成: 如果启用位姿头，将位姿信息融入处理

        参数:
            f_state: 状态特征 [B, state_size, dec_embed_dim]
            pos_state: 状态位置编码 [B, state_size, 2]
            f_img: 图像特征 [B, N, enc_embed_dim]
            pos_img: 图像位置编码 [B, N, 2]
            f_pose: 位姿特征 [B, 1, dec_embed_dim] (可选)
            pos_pose: 位姿位置编码 [B, 1, 2] (可选)

        返回:
            tuple: (状态特征序列, 图像特征序列)
            - 每个序列包含所有层的输出，用于多尺度处理
        """
        # === 初始化输出记录 ===
        # 记录每一层的输出，用于多尺度特征融合
        final_output = [(f_state, f_img)]  # 第0层：投影前的特征

        # === 状态特征维度检查 ===
        # 确保状态特征已经投影到解码器维度
        assert f_state.shape[-1] == self.dec_embed_dim

        # === 图像特征投影 ===
        # 将图像特征从编码器维度投影到解码器维度
        f_img = self.decoder_embed(f_img)
        # f_img: [B, N, dec_embed_dim]

        # === 位姿特征集成 ===
        if self.pose_head_flag:
            # 如果启用位姿头，将位姿特征添加到图像特征中
            assert f_pose is not None and pos_pose is not None

            # 在序列维度上拼接位姿特征和图像特征
            f_img = torch.cat([f_pose, f_img], dim=1)
            # f_img: [B, 1+N, dec_embed_dim] = [位姿token, 图像patches]

            # 相应地拼接位置编码
            pos_img = torch.cat([pos_pose, pos_img], dim=1)
            # pos_img: [B, 1+N, 2]

        # 记录投影后的特征
        final_output.append((f_state, f_img))

        # === 双解码器交替处理 ===
        # 这是CUT3R的核心：状态解码器和图像解码器的交替执行
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):

            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # === 训练时使用梯度检查点 ===
                # 状态解码器: 状态 attend to 图像
                f_state, _ = checkpoint(
                    blk_state,                  # 状态解码器块
                    *final_output[-1][::+1],    # (f_state, f_img) 正序
                    pos_state,                  # 状态位置编码
                    pos_img,                    # 图像位置编码
                    use_reentrant=not self.fixed_input_length,
                )

                # 图像解码器: 图像 attend to 状态
                f_img, _ = checkpoint(
                    blk_img,                    # 图像解码器块
                    *final_output[-1][::-1],    # (f_img, f_state) 逆序
                    pos_img,                    # 图像位置编码
                    pos_state,                  # 状态位置编码
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                # === 推理时直接前向传播 ===
                # 状态解码器: 状态作为查询，图像作为键值
                f_state, _ = blk_state(
                    *final_output[-1][::+1],    # (f_state, f_img)
                    pos_state,                  # 查询位置
                    pos_img                     # 键值位置
                )

                # 图像解码器: 图像作为查询，状态作为键值
                f_img, _ = blk_img(
                    *final_output[-1][::-1],    # (f_img, f_state)
                    pos_img,                    # 查询位置
                    pos_state                   # 键值位置
                )

            # 记录当前层的输出
            final_output.append((f_state, f_img))

        # === 输出处理 ===
        # 移除重复的第1层输出（与第0层相同）
        del final_output[1]

        # === 最终归一化 ===
        # 对最后一层的输出进行归一化
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),  # 状态特征归一化
            self.dec_norm(final_output[-1][1]),        # 图像特征归一化
        )

        # === 返回格式转换 ===
        # 将输出从 [(state, img), ...] 转换为 ([state, ...], [img, ...])
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape, **kwargs):
        """
        下游任务头部处理

        这个方法将解码器的输出转换为最终的预测结果，包括：
        - 3D点云坐标
        - 相机位姿
        - RGB颜色
        - 置信度评估

        参数:
            decout: 解码器输出特征列表
            img_shape: 图像形状信息
            **kwargs: 其他参数

        返回:
            预测结果字典，包含所有任务的输出
        """
        # 调用下游头部网络进行最终预测
        # 这个方法在父类CroCoNet中定义
        return self.downstream_head(decout, img_shape, **kwargs)

    def _get_img_level_feat(self, feat):
        """
        提取图像级别特征

        这个方法将补丁级别的特征聚合为图像级别的特征，
        用于位姿估计等需要全局信息的任务。

        参数:
            feat: 补丁特征 [B, N, D]

        返回:
            图像级特征 [B, 1, D] - 所有补丁特征的平均值
        """
        return torch.mean(feat, dim=1, keepdim=True)

    def _forward_encoder(self, views):
        """
        编码器前向传播

        这个方法执行CUT3R的编码阶段，包括：
        1. 编码所有视图的特征
        2. 初始化全局状态
        3. 初始化位姿记忆系统
        4. 准备递归处理所需的所有组件

        这是一个辅助方法，主要用于将编码和解码阶段分离，
        便于调试和模块化处理。

        参数:
            views: 输入视图列表

        返回:
            tuple: ((特征, 位置, 形状), (状态参数))
            - 第一个元组包含编码结果
            - 第二个元组包含初始化的状态参数
        """
        # === 步骤1: 编码所有视图 ===
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]  # 使用最深层特征

        # === 步骤2: 初始化状态和记忆 ===
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()

        # === 返回编码结果和状态参数 ===
        return (feat, pos, shape), (
            init_state_feat,    # 初始状态特征
            init_mem,           # 初始记忆
            state_feat,         # 当前状态特征
            state_pos,          # 状态位置编码
            mem,                # 当前记忆
        )

    def _forward_decoder_step(
        self,
        views,              # 视图列表
        i,                  # 当前帧索引
        feat_i,             # 当前帧特征
        pos_i,              # 当前帧位置编码
        shape_i,            # 当前帧形状
        init_state_feat,    # 初始状态特征
        init_mem,           # 初始记忆
        state_feat,         # 当前状态特征
        state_pos,          # 状态位置编码
        mem,                # 当前记忆
    ):
        """
        解码器单步前向传播

        这个方法执行单个时间步的解码处理，是递归推理的核心组件。
        它处理一帧输入并更新全局状态，生成该帧的预测结果。

        处理流程:
        1. 位姿特征处理（如果启用）
        2. 递归状态更新
        3. 位姿记忆更新
        4. 生成预测结果
        5. 状态更新控制

        参数:
            views: 视图列表
            i: 当前处理的帧索引
            feat_i: 当前帧的编码特征 [B, N, D]
            pos_i: 当前帧的位置编码 [B, N, 2]
            shape_i: 当前帧的形状信息
            init_state_feat: 初始状态特征（用于重置）
            init_mem: 初始记忆（用于重置）
            state_feat: 当前全局状态特征
            state_pos: 状态位置编码
            mem: 当前位姿记忆

        返回:
            tuple: (预测结果, (更新后的状态特征, 更新后的记忆))
        """
        # === 步骤1: 位姿特征处理 ===
        if self.pose_head_flag:
            # 提取图像级特征用于位姿估计
            global_img_feat_i = self._get_img_level_feat(feat_i)

            if i == 0:
                # 第一帧：使用可学习的位姿token初始化
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                # 后续帧：从位姿记忆中检索相关信息
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)

            # 位姿位置编码（使用-1作为特殊标识）
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            # 如果未启用位姿头，设置为None
            pose_feat_i = None
            pose_pos_i = None

        # === 步骤2: 递归状态更新 ===
        # 调用核心的递归更新方法
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,                     # 当前状态
            state_pos,                      # 状态位置
            feat_i,                         # 当前帧特征
            pos_i,                          # 当前帧位置
            pose_feat_i,                    # 位姿特征
            pose_pos_i,                     # 位姿位置
            init_state_feat,                # 初始状态
            img_mask=views[i]["img_mask"],  # 图像掩码
            reset_mask=views[i]["reset"],   # 重置掩码
            update=views[i].get("update", None),  # 更新掩码
        )

        # === 步骤3: 位姿记忆更新 ===
        # 提取位姿相关特征（解码器输出的第一个token）
        out_pose_feat_i = dec[-1][:, 0:1]

        # 更新位姿记忆库
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )

        # === 步骤4: 准备下游头部输入 ===
        # 选择多个尺度的特征用于最终预测
        head_input = [
            dec[0].float(),                              # 第0层（最浅）
            dec[self.dec_depth * 2 // 4][:, 1:].float(), # 1/2深度层，去掉位姿token
            dec[self.dec_depth * 3 // 4][:, 1:].float(), # 3/4深度层，去掉位姿token
            dec[self.dec_depth].float(),                 # 最深层
        ]

        # === 步骤5: 生成预测结果 ===
        res = self._downstream_head(head_input, shape_i, pos=pos_i)

        # === 步骤6: 状态更新控制 ===
        # 根据掩码决定是否更新状态
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)

        if update is not None:
            # 如果指定了更新掩码，只在满足条件时更新
            update_mask = img_mask & update
        else:
            # 否则根据图像掩码决定
            update_mask = img_mask

        # 转换为浮点掩码用于插值
        update_mask = update_mask[:, None, None].float()

        # === 步骤7: 应用状态更新 ===
        # 根据更新掩码选择性地更新全局状态
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )  # 更新全局状态

        # 同样更新位姿记忆
        mem = new_mem * update_mask + mem * (1 - update_mask)  # 更新局部状态

        # === 步骤8: 处理重置掩码 ===
        # 如果需要重置，恢复到初始状态
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)

        # === 返回结果 ===
        return res, (state_feat, mem)

    def _forward_impl(self, views, ret_state=False):
        """
        前向传播核心实现 - CUT3R的主要处理流程

        这是CUT3R模型的核心方法，实现了完整的连续3D感知流程：
        1. 编码所有视图的特征
        2. 初始化全局状态和记忆
        3. 逐帧递归处理和状态更新
        4. 生成每帧的预测结果
        5. 可选地返回状态信息用于调试或继续处理

        参数:
            views: 输入视图列表，每个视图包含图像、射线映射等信息
            ret_state: 是否返回状态信息（用于调试和在线推理）

        返回:
            如果ret_state=False: (预测结果列表, 视图列表)
            如果ret_state=True: (预测结果列表, 视图列表, 状态参数列表)
        """
        # === 步骤1: 编码所有视图 ===
        # 将所有输入视图编码为特征表示
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]  # 使用最后一层的特征（最深层）

        # === 步骤2: 初始化全局状态 ===
        # 使用第一个视图的特征初始化全局状态
        state_feat, state_pos = self._init_state(feat[0], pos[0])

        # === 步骤3: 初始化位姿记忆系统 ===
        # 扩展记忆库到批次维度
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)

        # 保存初始状态和记忆的副本（用于重置）
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()

        # === 步骤4: 状态跟踪初始化 ===
        # 记录所有状态参数，用于调试和分析
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]

        # === 步骤5: 逐帧处理 ===
        ress = []  # 存储每帧的预测结果

        for i in range(len(views)):
            # === 5.1: 获取当前帧特征 ===
            feat_i = feat[i]    # 当前帧的编码特征
            pos_i = pos[i]      # 当前帧的位置编码

            # === 5.2: 位姿特征处理 ===
            if self.pose_head_flag:
                # 获取图像级特征用于位姿估计
                global_img_feat_i = self._get_img_level_feat(feat_i)

                if i == 0:
                    # 第一帧：使用可学习的位姿token
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    # 后续帧：从记忆中检索位姿信息
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)

                # 位姿位置编码（使用特殊值-1标识）
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                # 如果未启用位姿头，设置为None
                pose_feat_i = None
                pose_pos_i = None

            # === 5.3: 递归状态更新 ===
            # 这是CUT3R的核心：递归地更新全局状态
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,                     # 当前全局状态
                state_pos,                      # 状态位置编码
                feat_i,                         # 当前帧特征
                pos_i,                          # 当前帧位置
                pose_feat_i,                    # 位姿特征
                pose_pos_i,                     # 位姿位置
                init_state_feat,                # 初始状态（用于重置）
                img_mask=views[i]["img_mask"],  # 图像掩码
                reset_mask=views[i]["reset"],   # 重置掩码
                update=views[i].get("update", None),  # 更新掩码
            )

            # === 5.4: 更新位姿记忆 ===
            if self.pose_head_flag:
                # 提取位姿相关特征（解码器输出的第一个token）
                out_pose_feat_i = dec[-1][:, 0:1]

                # 更新位姿记忆库
                new_mem = self.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )

            # === 5.5: 准备下游头部输入 ===
            # 确保解码器输出层数正确
            assert len(dec) == self.dec_depth + 1

            # 选择多个尺度的特征用于DPT头部
            head_input = [
                dec[0].float(),                              # 第0层（最浅）
                dec[self.dec_depth * 2 // 4][:, 1:].float(), # 1/2深度层
                dec[self.dec_depth * 3 // 4][:, 1:].float(), # 3/4深度层
                dec[self.dec_depth].float(),                 # 最深层
            ]
            # 注意：除第0层外，其他层去掉第一个token（位姿token）

            # === 5.6: 生成预测结果 ===
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)

            # === 5.7: 状态更新控制 ===
            # 根据掩码决定是否更新状态
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)

            if update is not None:
                # 如果指定了更新掩码，只在满足条件时更新
                update_mask = img_mask & update
            else:
                # 否则根据图像掩码决定
                update_mask = img_mask

            # 转换为浮点掩码用于插值
            update_mask = update_mask[:, None, None].float()

            # === 5.8: 应用状态更新 ===
            # 根据更新掩码选择性地更新全局状态
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)

            # 同样更新位姿记忆
            if self.pose_head_flag:
                mem = new_mem * update_mask + mem * (1 - update_mask)

            # === 5.9: 处理重置掩码 ===
            # 如果需要重置，恢复到初始状态
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                if self.pose_head_flag:
                    mem = init_mem * reset_mask + mem * (1 - reset_mask)

            # === 5.10: 记录状态参数 ===
            # 保存当前步骤的所有状态信息
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )

        # === 步骤6: 返回结果 ===
        if ret_state:
            # 返回预测结果、视图信息和完整的状态参数
            return ress, views, all_state_args
        else:
            # 只返回预测结果和视图信息
            return ress, views

    def forward(self, views, ret_state=False):
        """
        前向传播接口

        这是模型的主要入口点，提供标准的HuggingFace模型接口。
        它调用_forward_impl执行实际的前向传播，并将结果包装
        为标准的模型输出格式。

        参数:
            views: 输入视图列表
            ret_state: 是否返回状态信息

        返回:
            ARCroco3DStereoOutput: 标准化的模型输出
            如果ret_state=True，还会返回状态参数
        """
        if ret_state:
            # 需要返回状态信息的情况（用于在线推理）
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            # 标准推理情况
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem
    ):
        """
        单步推理接口 - 在线连续推理的核心

        这个方法实现了CUT3R的在线推理能力。它接受单个视图和
        之前的状态信息，输出当前帧的预测结果。这使得CUT3R
        能够进行真正的在线连续3D感知。

        使用场景:
        1. 实时视频流处理
        2. 长序列的增量处理
        3. 交互式3D重建
        4. 状态恢复和继续处理

        参数:
            view: 单个输入视图（包含图像、射线映射等）
            state_feat: 当前全局状态特征
            state_pos: 状态位置编码
            init_state_feat: 初始状态特征（用于重置）
            mem: 当前位姿记忆
            init_mem: 初始位姿记忆（用于重置）

        返回:
            tuple: (预测结果, 视图信息)
        """
        # === 步骤1: 数据预处理 ===
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []

        # 处理每个批次中的射线映射
        for j in range(batch_size):
            # 确保射线映射掩码为True
            assert view["ray_mask"][j]

            # 获取射线映射并调整维度
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)  # [1, 6, H, W]
            raymaps.append(raymap)

            # 获取真实形状信息
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        # 拼接所有射线映射和形状
        raymaps = torch.cat(raymaps, dim=0)  # [B, 6, H, W]
        shape = torch.cat(shapes, dim=0).to(raymaps.device)  # [B, 2]

        # === 步骤2: 编码射线映射 ===
        # 使用射线映射编码器处理几何信息
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes)
        feat_i = feat_ls[-1]  # 使用最深层特征
        pos_i = pos

        # === 步骤3: 位姿特征处理 ===
        if self.pose_head_flag:
            # 获取图像级特征
            global_img_feat_i = self._get_img_level_feat(feat_i)

            # 从记忆中检索位姿信息
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)

            # 设置位姿位置编码
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None

        # === 步骤4: 递归状态更新 ===
        # 使用传入的状态进行单步更新
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,                     # 使用传入的状态
            state_pos,                      # 状态位置编码
            feat_i,                         # 当前帧特征
            pos_i,                          # 当前帧位置
            pose_feat_i,                    # 位姿特征
            pose_pos_i,                     # 位姿位置
            init_state_feat,                # 初始状态
            img_mask=view["img_mask"],      # 图像掩码
            reset_mask=view["reset"],       # 重置掩码
            update=view.get("update", None), # 更新掩码
        )

        # === 步骤5: 更新位姿记忆 ===
        if self.pose_head_flag:
            # 提取位姿特征并更新记忆
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

        # === 步骤6: 生成预测结果 ===
        # 准备多尺度特征用于下游头部
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),                              # 第0层
            dec[self.dec_depth * 2 // 4][:, 1:].float(), # 1/2深度层
            dec[self.dec_depth * 3 // 4][:, 1:].float(), # 3/4深度层
            dec[self.dec_depth].float(),                 # 最深层
        ]

        # 调用下游头部生成最终预测
        res = self._downstream_head(head_input, shape, pos=pos_i)

        return res, view

    def forward_recurrent(self, views, device, ret_state=False):
        """
        递归前向传播 - 逐帧处理的在线推理模式

        这个方法实现了CUT3R的在线递归处理能力。与标准的forward方法
        不同，它逐帧处理输入，维护连续的状态，适用于长序列视频的
        实时处理。

        特点:
        1. 逐帧处理：每次只处理一个视图
        2. 状态持续：维护跨帧的全局状态和记忆
        3. 内存高效：避免同时加载所有帧
        4. 实时友好：支持流式处理

        参数:
            views: 视图列表，每个视图单独处理
            device: 计算设备
            ret_state: 是否返回状态信息

        返回:
            如果ret_state=False: (预测结果列表, 视图列表)
            如果ret_state=True: (预测结果列表, 视图列表, 状态参数列表)
        """
        ress = []               # 存储预测结果
        all_state_args = []     # 存储状态参数

        # === 逐帧处理循环 ===
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]

            # === 数据格式处理 ===
            # 将单帧数据重塑为批次格式
            img_mask = view["img_mask"].reshape(-1, batch_size)     # [1, B]
            ray_mask = view["ray_mask"].reshape(-1, batch_size)     # [1, B]
            imgs = view["img"].unsqueeze(0)                         # [1, B, C, H, W]
            ray_maps = view["ray_map"].unsqueeze(0)                 # [1, B, H, W, C]

            # 处理形状信息
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # [1, B, 2]

            # === 数据重塑 ===
            # 将数据从 [1, B, ...] 重塑为 [B, ...]
            imgs = imgs.view(-1, *imgs.shape[2:])           # [B, C, H, W]
            ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])  # [B, H, W, C]
            shapes = shapes.view(-1, 2).to(imgs.device)     # [B, 2]

            # 展平掩码
            img_masks_flat = img_mask.view(-1)              # [B]
            ray_masks_flat = ray_mask.view(-1)              # [B]

            # === 选择性编码 ===
            # 只编码有效的图像
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]

            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None

            # 只编码有效的射线映射
            ray_maps = ray_maps.permute(0, 3, 1, 2)         # [B, C, H, W]
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]

            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            # === 特征融合 ===
            shape = shapes
            if img_out is not None and ray_out is None:
                # 只有图像特征
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                # 只有射线映射特征
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                # 两种特征都有，进行加法融合
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                # 都没有，抛出异常
                raise NotImplementedError

            # === 状态初始化（仅第一帧） ===
            if i == 0:
                # 使用第一帧初始化全局状态和记忆
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

                # 记录初始状态
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )

            # === 位姿特征处理 ===
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)

                if i == 0:
                    # 第一帧使用位姿token
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    # 后续帧从记忆检索
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)

                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            # === 递归状态更新 ===
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
            )

            # === 更新位姿记忆 ===
            if self.pose_head_flag:
                out_pose_feat_i = dec[-1][:, 0:1]
                new_mem = self.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )

            # === 生成预测结果 ===
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)

            # === 状态更新控制 ===
            img_mask = view["img_mask"]
            update = view.get("update", None)

            if update is not None:
                update_mask = img_mask & update
            else:
                update_mask = img_mask

            update_mask = update_mask[:, None, None].float()

            # 更新全局状态和记忆
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
            if self.pose_head_flag:
                mem = new_mem * update_mask + mem * (1 - update_mask)

            # 处理重置掩码
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                if self.pose_head_flag:
                    mem = init_mem * reset_mask + mem * (1 - reset_mask)

            # 记录状态参数
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )

        # === 返回结果 ===
        if ret_state:
            return ress, views, all_state_args
        return ress, views


# ============================================================================
# 模型测试和使用示例
# ============================================================================

if __name__ == "__main__":
    """
    模型测试代码

    这部分代码展示了如何实例化和使用CUT3R模型。
    """
    # 打印模型的方法解析顺序（MRO）
    print(ARCroco3DStereo.mro())

    # 创建模型配置
    cfg = ARCroco3DStereoConfig(
        state_size=256,                     # 状态大小
        pos_embed="RoPE100",                # 位置编码类型
        rgb_head=True,                      # 启用RGB头
        pose_head=True,                     # 启用位姿头
        img_size=(224, 224),                # 图像尺寸
        head_type="linear",                 # 头部类型
        output_mode="pts3d+pose",           # 输出模式
        depth_mode=("exp", -inf, inf),      # 深度模式
        conf_mode=("exp", 1, inf),          # 置信度模式
        pose_mode=("exp", -inf, inf),       # 位姿模式
        enc_embed_dim=1024,                 # 编码器维度
        enc_depth=24,                       # 编码器深度
        enc_num_heads=16,                   # 编码器注意力头数
        dec_embed_dim=768,                  # 解码器维度
        dec_depth=12,                       # 解码器深度
        dec_num_heads=12,                   # 解码器注意力头数
    )

    # 实例化模型
    model = ARCroco3DStereo(cfg)
    print(f"模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================================
# 总结和设计思想
# ============================================================================

"""
CUT3R (ARCroco3DStereo) 模型架构总结

## 核心设计思想

1. **连续3D感知**
   - 全局状态管理：维护跨帧的场景理解
   - 递归状态更新：支持任意长度的视频序列
   - 记忆机制：LocalMemory存储历史位姿信息

2. **多模态融合**
   - RGB图像：提供视觉语义信息
   - 射线映射：提供几何约束信息
   - 双路编码器：分别处理不同模态的信息

3. **双解码器架构**
   - 状态解码器：更新全局状态表示
   - 图像解码器：细化局部图像特征
   - 交替处理：实现状态与图像的双向增强

4. **多任务学习**
   - 3D重建：预测每个像素的3D坐标
   - 位姿估计：预测相机的6DOF位姿
   - RGB重建：可选的颜色重建任务
   - 置信度估计：评估预测的可靠性

## 关键技术创新

1. **LocalMemory机制**
   - 可学习的记忆库，存储历史位姿信息
   - 支持长期依赖关系的建模
   - 通过注意力机制进行读写操作

2. **状态递归处理**
   - 维护全局状态表示
   - 支持任意长度的视频序列
   - 灵活的更新和重置机制

3. **在线推理能力**
   - inference_step：单步推理接口
   - forward_recurrent：递归前向传播
   - 支持实时视频流处理

4. **医疗场景优化**
   - 针对内窥镜图像的特殊处理
   - 处理非刚体组织变形
   - 强几何约束和置信度评估

## 模型优势

1. **连续性**：真正的连续3D感知，而非独立帧处理
2. **记忆性**：LocalMemory提供长期记忆能力
3. **实时性**：支持在线推理和流式处理
4. **鲁棒性**：多任务学习和置信度评估
5. **适应性**：针对医疗场景的专门优化

## 应用场景

1. **医疗内窥镜**：实时3D重建和导航
2. **机器人视觉**：连续环境理解
3. **AR/VR**：实时场景重建
4. **自动驾驶**：连续3D场景感知

这个架构通过精心设计的状态管理、多模态融合和递归处理，
实现了真正的连续3D感知能力，在医疗等关键应用中表现出色。
"""
