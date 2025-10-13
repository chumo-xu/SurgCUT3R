# CUT3R头部网络详细解释
# 本文件详细解释CUT3R项目中的各种头部网络

"""
头部网络(Head Networks)是模型的最后一层，负责将编码器-解码器的特征转换为最终的输出。

CUT3R支持多种头部网络：
1. Linear Head: 简单的线性投影头
2. DPT Head: 基于Dense Prediction Transformer的头部
3. 多任务头部: 同时输出3D点云、RGB、位姿等

输出类型：
- pts3d: 3D点云坐标
- conf: 置信度分数
- rgb: RGB颜色值
- pose: 相机位姿(7维：3D位置+4D四元数)
- desc: 局部特征描述子

头部网络的选择影响：
- 输出质量和精度
- 计算复杂度
- 内存使用量
- 训练稳定性
"""

from .linear_head import LinearPts3d, LinearPts3d_Desc, LinearPts3dPose
from .dpt_head import DPTPts3dPose


def head_factory(
    head_type,
    output_mode,
    net,
    has_conf=False,
    has_depth=False,
    has_rgb=False,
    has_pose_conf=False,
    has_pose=False,
):
    """
    头部网络工厂函数
    
    根据指定的参数创建相应的头部网络。这是一个工厂模式的实现，
    根据不同的配置组合返回不同类型的头部网络。
    
    参数:
        head_type: 头部类型 ("linear" 或 "dpt")
        output_mode: 输出模式 ("pts3d", "pts3d+pose", "pts3d+desc24"等)
        net: 网络实例，用于获取网络参数
        has_conf: 是否输出置信度
        has_depth: 是否输出深度信息
        has_rgb: 是否输出RGB颜色
        has_pose_conf: 是否输出位姿置信度
        has_pose: 是否输出位姿信息
        
    返回:
        相应的头部网络实例
        
    支持的组合：
    1. linear + pts3d: 基础3D点云预测
    2. linear + pts3d+pose: 3D点云 + 位姿预测
    3. linear + pts3d+desc: 3D点云 + 局部特征描述子
    4. dpt + pts3d+pose: 使用DPT的3D点云 + 位姿预测
    """
    if head_type == "linear" and output_mode == "pts3d":
        # 线性头部 + 基础3D点云输出
        return LinearPts3d(net, has_conf, has_depth, has_rgb, has_pose_conf)
    
    elif head_type == "linear" and output_mode == "pts3d+pose":
        # 线性头部 + 3D点云 + 位姿输出
        return LinearPts3dPose(net, has_conf, has_rgb, has_pose)
    
    elif head_type == "linear" and output_mode.startswith("pts3d+desc"):
        # 线性头部 + 3D点云 + 局部特征描述子
        local_feat_dim = int(output_mode[10:])  # 从"pts3d+desc24"中提取24
        return LinearPts3d_Desc(net, has_conf, has_depth, local_feat_dim)
    
    elif head_type == "dpt" and output_mode == "pts3d":
        # DPT头部 + 基础3D点云输出(未实现)
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
        # return create_dpt_head(net, has_conf=has_conf)
    
    elif head_type == "dpt" and output_mode == "pts3d+pose":
        # DPT头部 + 3D点云 + 位姿输出
        return DPTPts3dPose(net, has_conf, has_rgb, has_pose)
    
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")


class LinearHeadBase:
    """
    线性头部的基础类(概念性)
    
    线性头部的核心思想：
    1. 将解码器的输出特征通过简单的线性层转换
    2. 使用pixel_shuffle操作将补丁级特征转换为像素级输出
    3. 应用后处理函数得到最终结果
    
    优点：
    - 计算简单，速度快
    - 参数量少
    - 易于训练和调试
    
    缺点：
    - 表达能力有限
    - 对细节的处理能力较弱
    """
    pass


class DPTHeadBase:
    """
    DPT头部的基础类(概念性)
    
    DPT (Dense Prediction Transformer) 头部的核心思想：
    1. 使用多尺度特征融合
    2. 通过refinement网络逐步提升分辨率
    3. 更好地处理细节和边界
    
    优点：
    - 输出质量更高
    - 更好的细节保持
    - 多尺度信息融合
    
    缺点：
    - 计算复杂度高
    - 参数量大
    - 训练相对困难
    """
    pass


# 详细的工作流程说明
"""
头部网络的工作流程：

1. 输入处理：
   - 接收解码器的输出特征 (B, N, D)
   - N是补丁数量，D是特征维度

2. 特征转换：
   Linear Head: 直接线性投影
   DPT Head: 多尺度特征提取和融合

3. 空间重构：
   - 将补丁级特征转换为像素级输出
   - 使用pixel_shuffle或上采样操作

4. 后处理：
   - 应用激活函数(如exp, sigmoid等)
   - 处理不同的输出模式(深度、置信度等)

5. 多任务输出：
   - 3D点云: (B, 3, H, W) 每个像素的3D坐标
   - 置信度: (B, 1, H, W) 每个像素的可靠性
   - RGB: (B, 3, H, W) 重建的颜色
   - 位姿: (B, 7) 相机的位置和方向

具体的数据流：
解码器特征 (B, 196, 768) -> 线性投影 (B, 196, 3*16*16) -> 
pixel_shuffle -> 3D输出 (B, 3, 224, 224)

在您的配置中：
- head_type='dpt': 使用DPT头部
- output_mode='pts3d+pose': 输出3D点云和位姿
- rgb_head=True: 启用RGB输出
- pose_head=True: 启用位姿输出

这意味着模型会使用DPTPts3dPose头部，同时输出：
1. pts3d_in_self_view: 自视角的3D点云
2. pts3d_in_other_view: 其他视角的3D点云
3. conf_self: 自视角的置信度
4. conf: 跨视角的置信度
5. camera_pose: 相机位姿
6. rgb: RGB颜色(如果启用)

头部网络的选择策略：
1. 追求速度: 选择linear头部
2. 追求质量: 选择dpt头部
3. 多任务需求: 根据output_mode选择相应的头部
4. 内存限制: linear头部内存使用更少

训练考虑：
1. DPT头部需要更多的训练时间
2. 多任务头部需要平衡不同任务的损失权重
3. 位姿头部对数据质量要求较高
4. 置信度输出有助于提高模型的可靠性
"""


# 配置解析示例
"""
根据您的配置文件：

model: ARCroco3DStereo(ARCroco3DStereoConfig(
    ...
    rgb_head=True,           # 启用RGB头部
    pose_head=True,          # 启用位姿头部
    head_type='dpt',         # 使用DPT头部
    output_mode='pts3d+pose', # 输出3D点云和位姿
    ...
))

这个配置会创建一个DPTPts3dPose头部，具有以下特性：
1. 使用DPT架构进行高质量的密集预测
2. 同时输出3D点云和相机位姿
3. 包含RGB重建功能
4. 支持自视角和跨视角的置信度估计

数据流程：
输入多视角图像 -> 编码器 -> 解码器 -> DPT头部 -> 
{
    'pts3d_in_self_view': 自视角3D点云,
    'pts3d_in_other_view': 跨视角3D点云,
    'conf_self': 自视角置信度,
    'conf': 跨视角置信度,
    'camera_pose': 相机位姿,
    'rgb': RGB重建结果
}

这种配置特别适合：
- 医疗场景的3D重建
- 需要高精度位姿估计的应用
- 多视角几何约束的场景
- 需要置信度评估的安全关键应用
"""
