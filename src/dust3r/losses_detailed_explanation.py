"""
CUT3R项目Loss设计详细解释
=========================

这个文件是对CUT3R项目中losses.py的详细中文注释版本，旨在帮助理解整个损失函数系统的设计思路。

CUT3R是一个连续3D感知模型，其损失函数设计围绕以下核心概念：
1. 多视角几何一致性：确保不同视角预测的3D点在几何上一致
2. 置信度学习：通过学习置信度来加权不同预测的可靠性
3. 尺度不变性：处理深度估计中的尺度模糊问题
4. RGB重建：确保渲染的RGB图像与真实图像一致

关键术语解释：
- self view: 当前视角，即模型直接观察的视角
- cross view/other view: 其他视角，通过几何变换得到的视角
- pts3d_in_self_view: 在当前视角坐标系下的3D点
- pts3d_in_other_view: 变换到其他视角坐标系下的3D点
- norm_mode: 点云归一化模式，'?avg_dis'表示对非度量数据集使用平均距离归一化

配置文件中的损失函数设置：
训练时: ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)
测试时: Regr3DPose(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + 
        Regr3DPose_ScaleInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + 
        RGBLoss(L21)
"""

from copy import copy, deepcopy
import torch
import torch.nn as nn

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import (
    inv,        # 矩阵求逆函数
    geotrf,     # 几何变换函数，用于将3D点从一个坐标系变换到另一个坐标系
    normalize_pointcloud,       # 单个点云归一化
    normalize_pointcloud_group, # 多个点云组归一化
)
from dust3r.utils.geometry import (
    get_group_pointcloud_depth,         # 获取点云组的深度信息
    get_group_pointcloud_center_scale,  # 获取点云组的中心和尺度
    weighted_procrustes,                # 加权Procrustes对齐算法
)
from gsplat import rasterization  # 高斯点云渲染
import numpy as np
import lpips  # 感知损失
from dust3r.utils.camera import (
    pose_encoding_to_camera,    # 位姿编码转相机矩阵
    camera_to_pose_encoding,    # 相机矩阵转位姿编码
    relative_pose_absT_quatR,   # 相对位姿计算
)


def Sum(*losses_and_masks):
    """
    损失函数求和工具函数
    
    这个函数用于将多个损失函数及其对应的掩码进行求和。
    它能智能地处理两种情况：
    1. 像素级损失：返回每个像素的损失值
    2. 全局损失：返回标量损失值
    
    参数:
        *losses_and_masks: 多个(loss, mask)元组
        
    返回:
        如果是像素级损失，返回原始的losses_and_masks
        如果是全局损失，返回求和后的标量损失
    """
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # 像素级损失：每个像素都有对应的损失值
        # 这种情况下直接返回所有损失和掩码，让上层处理
        return losses_and_masks
    else:
        # 全局损失：整个图像/批次只有一个损失值
        # 将所有损失相加得到最终损失
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    """
    损失函数基类
    
    所有具体的损失函数都继承自这个基类。
    提供了统一的reduction参数处理机制。
    
    参数:
        reduction: 损失聚合方式
            - "mean": 取平均值
            - "sum": 求和
            - "none": 不聚合，返回每个元素的损失
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    """
    L范数损失的抽象基类
    
    这是一个抽象类，定义了L范数损失的通用框架。
    具体的距离计算由子类实现。
    
    支持的输入形状：
    - 2D或更高维度的张量
    - 最后一个维度的大小必须在1-3之间（对应1D、2D或3D点）
    """
    
    def forward(self, a, b):
        """
        前向传播
        
        参数:
            a, b: 需要计算距离的两个张量，形状必须相同
            
        返回:
            根据reduction模式返回相应的损失值
        """
        assert (
            a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3
        ), f"Bad shape = {a.shape}"
        
        # 计算距离（由子类实现）
        dist = self.distance(a, b)
        
        # 根据reduction模式处理结果
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b):
        """抽象方法：计算两个张量之间的距离"""
        raise NotImplementedError()


class L21Loss(LLoss):
    """
    L2范数损失（欧几里得距离）
    
    计算两个3D点之间的欧几里得距离。
    这是3D点回归任务中最常用的损失函数。
    
    数学公式: ||a - b||_2
    """
    
    def distance(self, a, b):
        """计算欧几里得距离"""
        return torch.norm(a - b, dim=-1)  # 沿最后一个维度计算L2范数


# 创建全局L21损失实例，供其他模块使用
L21 = L21Loss()


class MSELoss(LLoss):
    """
    均方误差损失
    
    计算两个张量之间的平方误差。
    相比L21Loss，MSELoss对大误差更敏感。
    
    数学公式: (a - b)^2
    """
    
    def distance(self, a, b):
        """计算平方误差"""
        return (a - b) ** 2


# 创建全局MSE损失实例
MSE = MSELoss()


class Criterion(nn.Module):
    """
    损失函数包装器
    
    这个类将BaseCriterion包装起来，提供额外的功能：
    1. 类型检查：确保传入的是有效的损失函数
    2. 深拷贝：避免多个实例共享状态
    3. Reduction模式切换：支持动态改变聚合方式
    """
    
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(
            criterion, BaseCriterion
        ), f"{criterion} is not a proper criterion!"
        self.criterion = copy(criterion)

    def get_name(self):
        """获取损失函数的名称"""
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        """
        创建一个新的损失函数实例，使用指定的reduction模式
        
        这个方法用于在需要像素级损失时临时改变reduction模式。
        例如，ConfLoss需要像素级的损失值来计算置信度加权。
        
        参数:
            mode: 新的reduction模式
            
        返回:
            新的损失函数实例
        """
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # 设置新的reduction模式
            loss = loss._loss2  # 处理链式损失（MultiLoss）
        return res


class MultiLoss(nn.Module):
    """
    多损失函数组合器

    这是CUT3R损失系统的核心设计，允许通过简单的数学运算符组合多个损失函数：

    使用示例:
        loss = MyLoss1() + 0.1*MyLoss2() + 0.5*MyLoss3()

    设计特点:
    1. 链式结构：通过_loss2属性形成链表结构
    2. 权重支持：通过_alpha属性支持损失加权
    3. 自动追踪：自动记录每个子损失的值用于监控
    4. 运算符重载：支持+和*运算符的自然语法

    这种设计让损失函数的组合变得非常直观和灵活。
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1      # 当前损失的权重系数
        self._loss2 = None   # 链表中的下一个损失函数

    def compute_loss(self, *args, **kwargs):
        """抽象方法：计算具体的损失值，由子类实现"""
        raise NotImplementedError()

    def get_name(self):
        """抽象方法：获取损失函数名称，由子类实现"""
        raise NotImplementedError()

    def __mul__(self, alpha):
        """
        重载乘法运算符，支持损失加权

        例如: 0.5 * MyLoss() 会创建一个权重为0.5的损失函数

        参数:
            alpha: 权重系数

        返回:
            新的加权损失函数实例
        """
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # 支持右乘：alpha * loss

    def __add__(self, loss2):
        """
        重载加法运算符，支持损失函数组合

        例如: Loss1() + Loss2() 会创建一个组合损失函数

        参数:
            loss2: 要添加的另一个损失函数

        返回:
            组合后的损失函数
        """
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # 找到链表的末尾
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2  # 将新损失添加到链表末尾
        return res

    def __repr__(self):
        """字符串表示，用于调试和日志"""
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        """
        前向传播：计算组合损失

        这个方法会：
        1. 计算当前损失函数的值
        2. 应用权重系数
        3. 递归计算链表中其他损失函数的值
        4. 收集所有子损失的详细信息用于监控

        返回:
            (total_loss, details): 总损失和详细信息字典
        """
        # 计算当前损失
        loss = self.compute_loss(*args, **kwargs)

        # 处理返回值：可能是单个损失值或(损失值, 详细信息)元组
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            # 标量损失：创建详细信息字典
            details = {self.get_name(): float(loss)}
        else:
            # 张量损失：不创建详细信息
            details = {}

        # 应用权重
        loss = loss * self._alpha

        # 递归处理链表中的下一个损失函数
        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2  # 合并详细信息字典

        return loss, details


class SSIM(nn.Module):
    """
    结构相似性指数损失 (Structural Similarity Index Loss)

    SSIM是一种感知质量度量，比简单的像素级损失更符合人类视觉感知。
    它考虑了图像的亮度、对比度和结构信息。

    SSIM的计算涉及：
    1. 亮度比较：比较两幅图像的平均亮度
    2. 对比度比较：比较两幅图像的方差
    3. 结构比较：比较两幅图像去除亮度和对比度后的相关性

    数学公式:
    SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))

    其中:
    - μx, μy: 图像x和y的均值
    - σx², σy²: 图像x和y的方差
    - σxy: 图像x和y的协方差
    - C1, C2: 稳定常数
    """

    def __init__(self):
        super(SSIM, self).__init__()
        # 使用3x3平均池化来计算局部统计量
        self.mu_x_pool = nn.AvgPool2d(3, 1)      # 计算x的局部均值
        self.mu_y_pool = nn.AvgPool2d(3, 1)      # 计算y的局部均值
        self.sig_x_pool = nn.AvgPool2d(3, 1)     # 计算x的局部方差
        self.sig_y_pool = nn.AvgPool2d(3, 1)     # 计算y的局部方差
        self.sig_xy_pool = nn.AvgPool2d(3, 1)    # 计算x和y的局部协方差

        # 反射填充，保持输出尺寸与输入相同
        self.refl = nn.ReflectionPad2d(1)

        # SSIM稳定常数（基于像素值范围[0,1]）
        self.C1 = 0.01**2  # 亮度比较的稳定常数
        self.C2 = 0.03**2  # 对比度比较的稳定常数

    def forward(self, x, y):
        """
        计算SSIM损失

        参数:
            x, y: 输入图像张量，形状为(B, C, H, W)

        返回:
            SSIM损失，值域[0, 1]，0表示完全相似，1表示完全不同
        """
        # 边界填充
        x = self.refl(x)
        y = self.refl(y)

        # 计算局部均值
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # 计算局部方差和协方差
        sigma_x = self.sig_x_pool(x**2) - mu_x**2      # Var(X) = E[X²] - E[X]²
        sigma_y = self.sig_y_pool(y**2) - mu_y**2      # Var(Y) = E[Y²] - E[Y]²
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y  # Cov(X,Y) = E[XY] - E[X]E[Y]

        # 计算SSIM
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)  # 分子
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)  # 分母

        # 转换为损失：(1 - SSIM) / 2，并限制在[0,1]范围内
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class RGBLoss(Criterion, MultiLoss):
    """
    RGB图像重建损失

    这个损失函数确保模型预测的RGB图像与真实图像一致。
    在CUT3R中，模型不仅要预测3D几何，还要能够渲染出逼真的RGB图像。

    设计思路:
    1. 继承Criterion：包装基础损失函数（如MSE或L21）
    2. 继承MultiLoss：支持与其他损失函数组合
    3. 集成SSIM：可选地使用结构相似性损失

    在配置文件中的使用:
    - 训练时: RGBLoss(MSE) - 使用MSE作为基础损失
    - 测试时: RGBLoss(L21) - 使用L21作为基础损失
    """

    def __init__(self, criterion):
        super().__init__(criterion)
        self.ssim = SSIM()  # SSIM损失，虽然初始化了但在当前实现中未使用

    def img_loss(self, a, b):
        """计算两幅图像之间的损失"""
        return self.criterion(a, b)

    def compute_loss(self, gts, preds, **kw):
        """
        计算RGB重建损失

        参数:
            gts: 真实数据列表，每个元素包含"img"字段
            preds: 预测数据列表，每个元素包含"rgb"字段
            **kw: 其他关键字参数

        返回:
            (rgb_loss, details): RGB损失和详细信息
        """
        # 提取真实RGB图像：从(B,C,H,W)转换为(B,H,W,C)
        gt_rgbs = [gt["img"].permute(0, 2, 3, 1) for gt in gts]

        # 提取预测RGB图像
        pred_rgbs = [pred["rgb"] for pred in preds]

        # 计算每个视图的RGB损失
        ls = [
            self.img_loss(pred_rgb, gt_rgb)
            for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs)
        ]

        # 收集详细信息用于监控和可视化
        details = {}
        self_name = type(self).__name__
        for i, l in enumerate(ls):
            details[self_name + f"_rgb/{i+1}"] = float(l)  # 记录每个视图的损失
            details[f"pred_rgb_{i+1}"] = pred_rgbs[i]      # 保存预测图像用于可视化

        # 计算平均RGB损失
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details


class DepthScaleShiftInvLoss(BaseCriterion):
    """
    深度尺度平移不变损失

    这个损失函数解决了深度估计中的尺度和平移模糊问题。
    在单目深度估计中，由于缺乏绝对尺度信息，预测的深度往往存在
    全局的尺度和平移偏差。

    核心思想:
    1. 尺度不变：通过归一化消除全局尺度差异
    2. 平移不变：通过去中心化消除全局平移偏差
    3. 只在有效区域计算：使用mask排除无效像素

    数学原理:
    对于预测深度pred和真实深度gt，分别进行归一化：
    pred_norm = (pred - mean(pred)) / mean(|pred - mean(pred)|)
    gt_norm = (gt - mean(gt)) / mean(|gt - mean(gt)|)
    然后计算归一化后的L1距离。

    适用场景:
    - 相对深度监督：只有深度的相对关系，没有绝对尺度
    - 单目深度估计：存在尺度模糊问题
    - 深度补全：部分区域的深度信息缺失
    """

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        """
        前向传播

        参数:
            pred: 预测深度，形状(B, H, W)
            gt: 真实深度，形状(B, H, W)
            mask: 有效区域掩码，形状(B, H, W)

        返回:
            尺度平移不变损失
        """
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"

        # 计算归一化距离
        dist = self.distance(pred, gt, mask)

        # 根据reduction模式处理结果
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x, mask):
        """
        对深度图进行尺度平移归一化

        步骤:
        1. 提取有效像素
        2. 计算每个样本的均值（平移）
        3. 去中心化
        4. 计算每个样本的平均绝对偏差（尺度）
        5. 归一化

        参数:
            x: 输入深度图，形状(B, H, W)
            mask: 有效区域掩码，形状(B, H, W)

        返回:
            归一化后的深度图
        """
        # 提取所有有效像素
        x_valid = x[mask]

        # 按批次分割有效像素
        splits = mask.sum(dim=(1, 2)).tolist()  # 每个样本的有效像素数
        x_valid_list = torch.split(x_valid, splits)

        # 计算每个样本的均值（用于去中心化）
        shift = [x.mean() for x in x_valid_list]

        # 去中心化
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]

        # 计算每个样本的平均绝对偏差（用于尺度归一化）
        scale = [x.abs().mean() for x in x_valid_centered]

        # 转换为张量
        scale = torch.stack(scale)
        shift = torch.stack(shift)

        # 应用归一化：(x - shift) / scale
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(self, pred, gt, mask):
        """
        计算归一化后的L1距离

        参数:
            pred: 预测深度
            gt: 真实深度
            mask: 有效区域掩码

        返回:
            有效区域内的L1距离
        """
        # 分别归一化预测和真实深度
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)

        # 计算有效区域内的L1距离
        return torch.abs((pred - gt)[mask])


class ScaleInvLoss(BaseCriterion):
    """
    尺度不变损失（用于3D点云）

    与DepthScaleShiftInvLoss类似，但专门用于3D点云数据。
    这个损失函数通过归一化消除点云的全局尺度差异。

    核心思想:
    1. 计算每个点云的平均范数作为尺度因子
    2. 用尺度因子归一化点云
    3. 计算归一化后点云的L2距离

    数学原理:
    scale_pred = mean(||pred_points||)
    scale_gt = mean(||gt_points||)
    pred_norm = pred / scale_pred
    gt_norm = gt / scale_gt
    loss = ||pred_norm - gt_norm||

    适用场景:
    - 单视图3D重建：缺乏绝对尺度信息
    - 点云配准：需要尺度不变的匹配
    - 3D形状比较：关注形状而非绝对大小
    """

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        """
        前向传播

        参数:
            pred: 预测3D点，形状(B, H, W, 3)
            gt: 真实3D点，形状(B, H, W, 3)
            mask: 有效区域掩码，形状(B, H, W)

        返回:
            尺度不变损失
        """
        assert pred.shape == gt.shape and pred.ndim == 4, f"Bad shape = {pred.shape}"

        # 计算尺度不变距离
        dist = self.distance(pred, gt, mask)

        # 根据reduction模式处理结果
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, pred, gt, mask):
        """
        计算尺度不变距离

        步骤:
        1. 计算每个样本中有效点的平均范数作为尺度因子
        2. 用尺度因子归一化点云
        3. 计算归一化后的L2距离

        参数:
            pred: 预测3D点
            gt: 真实3D点
            mask: 有效区域掩码

        返回:
            有效区域内的尺度不变L2距离
        """
        # 计算预测点云的尺度因子
        # torch.norm(pred, dim=-1): 计算每个点的L2范数
        # * mask: 只考虑有效点
        # .sum(dim=(1, 2)): 对H和W维度求和
        # / mask.sum(dim=(1, 2)): 除以有效点数量，得到平均范数
        pred_norm_factor = (torch.norm(pred, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)

        # 计算真实点云的尺度因子
        gt_norm_factor = (torch.norm(gt, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)

        # 归一化点云
        pred = pred / pred_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        gt = gt / gt_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)

        # 计算有效区域内的L2距离
        return torch.norm(pred - gt, dim=-1)[mask]


class Regr3DPose(Criterion, MultiLoss):
    """
    3D点回归和位姿估计的核心损失函数

    这是CUT3R最重要的损失函数，负责确保多视角3D点预测的几何一致性。

    核心概念:
    1. 非对称损失：view1作为锚点视角（anchor view）
    2. Self view loss：当前视角下的3D点回归损失
    3. Cross view loss：跨视角几何一致性损失
    4. 位姿损失：相机位姿估计损失

    几何原理:
    对于两个视角的图像，模型预测：
    - P1 = pts3d_in_self_view: 在view1坐标系下的3D点
    - P2 = pts3d_in_other_view: 变换到view2坐标系下的3D点
    - RT: 从view1到view2的相机位姿变换

    一致性约束:
    - Self view: 预测的P1应该与真实的3D点一致
    - Cross view: RT(P1)应该与P2一致

    数学表示:
    loss1 = ||pred_P1 - gt_P1||  (self view loss)
    loss2 = ||RT(pred_P1) - pred_P2||  (cross view loss)

    在配置文件中的使用:
    - 训练时: Regr3DPoseBatchList(L21, norm_mode='?avg_dis')
    - 测试时: Regr3DPose(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)
    """

    def __init__(
        self,
        criterion,                    # 基础损失函数（如L21）
        norm_mode="?avg_dis",        # 归一化模式
        gt_scale=False,              # 是否使用GT尺度
        sky_loss_value=2,            # 天空区域的损失值
        max_metric_scale=False,      # 最大度量尺度限制
    ):
        super().__init__(criterion)

        # 解析归一化模式
        if norm_mode.startswith("?"):
            # "?"前缀表示：对度量尺度数据集不进行归一化
            # 这是因为度量数据集已经有正确的绝对尺度
            self.norm_all = False
            self.norm_mode = norm_mode[1:]  # 去掉"?"前缀
        else:
            # 对所有数据集都进行归一化
            self.norm_all = True
            self.norm_mode = norm_mode

        self.gt_scale = gt_scale                    # 是否强制使用GT尺度
        self.sky_loss_value = sky_loss_value        # 天空区域的特殊损失值
        self.max_metric_scale = max_metric_scale    # 度量尺度的最大值限制

    def get_norm_factor_point_cloud(
        self, pts_self, pts_cross, valids, conf_self, conf_cross, norm_self_only=False
    ):
        """
        计算点云归一化因子

        这个方法根据norm_mode计算点云的归一化因子，用于消除尺度差异。

        参数:
            pts_self: 自视角3D点列表
            pts_cross: 跨视角3D点列表
            valids: 有效性掩码列表
            conf_self: 自视角置信度列表
            conf_cross: 跨视角置信度列表
            norm_self_only: 是否只使用自视角点进行归一化

        返回:
            归一化因子张量
        """
        if norm_self_only:
            # 只使用自视角点计算归一化因子
            norm_factor = normalize_pointcloud_group(
                pts_self, self.norm_mode, valids, conf_self, ret_factor_only=True
            )
        else:
            # 使用自视角和跨视角点联合计算归一化因子
            # 将自视角和跨视角点连接起来
            pts = [torch.cat([x, y], dim=2) for x, y in zip(pts_self, pts_cross)]
            valids = [torch.cat([x, x], dim=2) for x in valids]
            confs = [torch.cat([x, y], dim=2) for x, y in zip(conf_self, conf_cross)]

            norm_factor = normalize_pointcloud_group(
                pts, self.norm_mode, valids, confs, ret_factor_only=True
            )
        return norm_factor

    def get_norm_factor_poses(self, gt_trans, pr_trans, not_metric_mask):
        """
        计算位姿归一化因子

        对于相机位姿，也需要进行归一化以保持与点云的尺度一致性。

        参数:
            gt_trans: 真实位姿平移部分列表
            pr_trans: 预测位姿平移部分列表
            not_metric_mask: 非度量数据掩码

        返回:
            (norm_factor_gt, norm_factor_pr): GT和预测的归一化因子
        """
        # 计算GT位姿的归一化因子
        if self.norm_mode and not self.gt_scale:
            # 将平移向量reshape为点云格式进行归一化
            gt_trans = [x[:, None, None, :].clone() for x in gt_trans]
            valids = [torch.ones_like(x[..., 0], dtype=torch.bool) for x in gt_trans]
            norm_factor_gt = (
                normalize_pointcloud_group(
                    gt_trans,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
        else:
            # 不进行归一化，使用单位因子
            norm_factor_gt = torch.ones(
                len(gt_trans), dtype=gt_trans[0].dtype, device=gt_trans[0].device
            )

        # 预测位姿的归一化因子初始化为GT的因子
        norm_factor_pr = norm_factor_gt.clone()

        # 对非度量数据，单独计算预测位姿的归一化因子
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            pr_trans_not_metric = [
                x[not_metric_mask][:, None, None, :].clone() for x in pr_trans
            ]
            valids = [
                torch.ones_like(x[..., 0], dtype=torch.bool)
                for x in pr_trans_not_metric
            ]
            norm_factor_pr_not_metric = (
                normalize_pointcloud_group(
                    pr_trans_not_metric,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        return norm_factor_gt, norm_factor_pr

    def compute_pose_loss(self, gt_poses, pred_poses, masks=None):
        """
        计算位姿损失

        位姿损失包括平移损失和旋转损失两部分。
        位姿用(translation, quaternion)格式表示。

        参数:
            gt_poses: 真实位姿列表，每个元素为(Bx3, Bx4)元组
            pred_poses: 预测位姿列表，每个元素为(Bx3, Bx4)元组
            masks: 有效性掩码，形状为B或None

        返回:
            位姿损失（平移损失 + 旋转损失）
        """
        # 提取平移和旋转部分
        gt_trans = torch.stack([gt[0] for gt in gt_poses], dim=1)    # BxNx3
        gt_quats = torch.stack([gt[1] for gt in gt_poses], dim=1)    # BxNx4
        pred_trans = torch.stack([pr[0] for pr in pred_poses], dim=1) # BxNx3
        pred_quats = torch.stack([pr[1] for pr in pred_poses], dim=1) # BxNx4

        if masks is None:
            # 没有掩码：计算所有样本的损失
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1).mean()  # 平移损失
                + torch.norm(pred_quats - gt_quats, dim=-1).mean()  # 旋转损失
            )
        else:
            # 有掩码：只计算有效样本的损失
            # 健壮性检查：处理空掩码的情况
            is_empty = False
            if isinstance(masks, torch.Tensor):
                if not torch.any(masks):
                    is_empty = True
            elif not any(masks):
                is_empty = True

            if is_empty:
                # 如果没有有效样本，返回零损失
                return torch.tensor(0.0, device=pred_trans.device)

            # 计算有效样本的位姿损失
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1)[masks].mean()  # 平移损失
                + torch.norm(pred_quats - gt_quats, dim=-1)[masks].mean()  # 旋转损失
            )

        return pose_loss

    # 注意：get_all_pts3d方法非常复杂，包含了大量的3D几何处理逻辑
    # 由于篇幅限制，这里不展开详细解释，但它的主要功能是：
    # 1. 坐标系变换：将所有3D点变换到统一的坐标系
    # 2. 点云归一化：根据norm_mode进行尺度归一化
    # 3. 掩码处理：处理有效性掩码和天空掩码
    # 4. 位姿处理：处理相机位姿的编码和归一化
    # 具体实现请参考原始代码中的get_all_pts3d方法


class ConfLoss(MultiLoss):
    """
    置信度加权损失 - CUT3R的核心创新之一

    这个损失函数实现了自适应的置信度学习机制，让模型能够：
    1. 学习预测每个像素的可靠性
    2. 根据置信度自适应地加权损失
    3. 在不确定区域降低损失权重，在确定区域增加损失权重

    核心原理:
    传统损失函数对所有像素一视同仁，但实际上不同像素的预测难度不同。
    ConfLoss让模型学习一个置信度图，用来表示每个像素预测的可靠性。

    数学公式:
    对于像素级损失pixel_loss和置信度conf：
    conf_loss = pixel_loss * conf - alpha * log(conf)

    直觉理解:
    - 高置信度(conf小，如0.1): 损失被放大，模型必须预测得很准确
    - 低置信度(conf大，如10): 损失被缩小，模型可以"承认"不确定性
    - log(conf)项防止模型总是预测低置信度来逃避损失

    在配置文件中的使用:
    ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2)
    - 基础损失: Regr3DPoseBatchList(L21) - 3D点回归损失
    - alpha=0.2: 置信度正则化强度

    这种设计让模型能够：
    1. 在纹理丰富、几何清晰的区域预测高置信度
    2. 在纹理缺乏、遮挡严重的区域预测低置信度
    3. 自动平衡不同难度区域的学习
    """

    def __init__(self, pixel_loss, alpha=1):
        """
        初始化置信度损失

        参数:
            pixel_loss: 像素级损失函数（如Regr3DPose）
            alpha: 置信度正则化系数，控制log(conf)项的权重
        """
        super().__init__()
        assert alpha > 0, "alpha必须为正数"
        self.alpha = alpha
        # 将像素级损失设置为"none"模式，获取每个像素的损失值
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self):
        """获取损失函数名称"""
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        """
        获取置信度和其对数

        这个方法可以被子类重写来实现不同的置信度处理方式。

        参数:
            x: 置信度张量

        返回:
            (conf, log_conf): 置信度和其对数
        """
        return x, torch.log(x)

    def compute_loss(self, gts, preds, **kw):
        """
        计算置信度加权损失

        这个方法的核心逻辑：
        1. 计算像素级损失
        2. 提取对应的置信度
        3. 应用置信度加权公式
        4. 聚合所有损失

        参数:
            gts: 真实数据列表
            preds: 预测数据列表
            **kw: 其他关键字参数

        返回:
            (final_loss, details): 最终损失和详细信息
        """
        # 1. 计算像素级损失
        losses_and_masks, details = self.pixel_loss(gts, preds, **kw)

        # 2. 获取损失的元信息
        if "is_self" in details and "img_ids" in details:
            is_self = details["is_self"]    # 是否为自视角损失
            img_ids = details["img_ids"]    # 图像索引
        else:
            # 如果没有元信息，假设都是跨视角损失
            is_self = [False] * len(losses_and_masks)
            img_ids = list(range(len(losses_and_masks)))

        # 3. 对每个损失应用置信度加权
        conf_losses = []

        for i in range(len(losses_and_masks)):
            pred = preds[img_ids[i]]

            # 根据是否为自视角选择对应的置信度
            conf_key = "conf_self" if is_self[i] else "conf"

            if not is_self[i]:
                # 跨视角损失：需要排除camera_only的样本
                camera_only = gts[0]["camera_only"]
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][~camera_only][losses_and_masks[i][1]]
                )
            else:
                # 自视角损失：直接使用置信度
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][losses_and_masks[i][1]]
                )

            # 4. 应用置信度加权公式
            # conf_loss = pixel_loss * conf - alpha * log(conf)
            conf_loss = losses_and_masks[i][0] * conf - self.alpha * log_conf

            # 聚合为标量损失
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)

            # 记录详细信息用于监控
            if is_self[i]:
                details[self.get_name() + f"_conf_loss_self/{img_ids[i]+1}"] = float(conf_loss)
            else:
                details[self.get_name() + f"_conf_loss/{img_ids[i]+1}"] = float(conf_loss)

        # 清理临时的元信息
        details.pop("is_self", None)
        details.pop("img_ids", None)

        # 5. 计算最终损失
        # 平均所有置信度损失，乘以2.0作为缩放因子
        final_loss = sum(conf_losses) / len(conf_losses) * 2.0

        # 6. 添加位姿损失（如果存在）
        if "pose_loss" in details:
            # 位姿损失直接加到最终损失中
            final_loss = final_loss + details["pose_loss"]

        # 7. 添加尺度损失（如果存在）
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]

        return final_loss, details


class Regr3DPoseBatchList(Regr3DPose):
    """
    批处理版本的3D位姿回归损失

    这是Regr3DPose的优化版本，专门设计用于高效处理批量数据。
    相比基础版本，它提供了更好的内存效率和计算性能。

    核心改进：
    1. 批处理优化：将多个样本的损失计算合并，减少循环开销
    2. 内存优化：更高效的张量操作，减少内存碎片
    3. 多种损失类型支持：根据数据特性选择不同的损失函数

    适用场景：
    - 大批量训练：当batch_size较大时性能优势明显
    - 混合数据集：同时处理不同类型的数据（深度、单视图、多视图）
    - 生产环境：需要高效推理的应用场景

    在配置文件中的使用：
    train_criterion: ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2)
    """

    def __init__(
        self,
        criterion,                    # 基础损失函数
        norm_mode="?avg_dis",        # 归一化模式
        gt_scale=False,              # 是否使用GT尺度
        sky_loss_value=2,            # 天空区域损失值
        max_metric_scale=False,      # 最大度量尺度限制
    ):
        # 继承父类的所有功能
        super().__init__(
            criterion, norm_mode, gt_scale, sky_loss_value, max_metric_scale
        )

        # 添加专门的损失函数用于不同数据类型
        self.depth_only_criterion = DepthScaleShiftInvLoss()  # 仅深度数据的损失
        self.single_view_criterion = ScaleInvLoss()           # 单视图数据的损失

    def reorg(self, ls_b, masks_b):
        """
        重新组织批处理损失数据

        这个方法将批处理计算的损失重新组织为按视图分组的格式，
        以便与原始的损失计算流程兼容。

        参数:
            ls_b: 批处理损失列表，每个元素对应一个批次
            masks_b: 批处理掩码列表，每个元素对应一个批次

        返回:
            按视图重新组织的损失列表

        工作原理:
        1. 计算每个掩码的有效像素数量
        2. 根据像素数量分割损失张量
        3. 按视图重新组合损失数据
        """
        # 计算每个批次中每个样本的有效像素数量
        ids_split = [mask.sum(dim=(1, 2)) for mask in masks_b]

        # 初始化按视图分组的损失列表
        ls = [[] for _ in range(len(masks_b[0]))]  # masks_b[0]的长度是batch_size

        # 重新组织损失数据
        for i in range(len(ls_b)):
            # 根据有效像素数量分割当前批次的损失
            ls_splitted_i = torch.split(ls_b[i], ids_split[i].tolist())

            # 将分割后的损失分配到对应的视图
            for j in range(len(masks_b[0])):
                ls[j].append(ls_splitted_i[j])

        # 连接同一视图的所有损失
        ls = [torch.cat(l) for l in ls]
        return ls

    def compute_loss(self, gts, preds, **kw):
        """
        计算批处理版本的3D位姿损失

        这个方法是BatchList版本的核心，它能够：
        1. 高效处理不同类型的数据（深度、单视图、多视图）
        2. 根据数据特性选择合适的损失函数
        3. 批量计算损失以提高效率

        参数:
            gts: 真实数据列表
            preds: 预测数据列表
            **kw: 其他关键字参数

        返回:
            (total_loss, details): 总损失和详细信息
        """
        # 1. 获取所有3D点和位姿数据（继承自父类）
        (
            gt_pts_self,      # 真实自视角3D点
            gt_pts_cross,     # 真实跨视角3D点
            pred_pts_self,    # 预测自视角3D点
            pred_pts_cross,   # 预测跨视角3D点
            gt_poses,         # 真实位姿
            pr_poses,         # 预测位姿
            masks,            # 有效性掩码
            skys,             # 天空掩码
            pose_masks,       # 位姿掩码
            monitoring,       # 监控信息
        ) = self.get_all_pts3d(gts, preds, **kw)

        # 2. 处理天空区域
        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            # 将天空区域包含在有效掩码中
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        # 3. 获取数据类型标识
        camera_only = gts[0]["camera_only"]    # 仅相机参数的样本
        depth_only = gts[0]["depth_only"]      # 仅深度数据的样本
        single_view = gts[0]["single_view"]    # 单视图样本
        is_metric = gts[0]["is_metric"]        # 度量数据标识

        # 4. 计算自视角损失（批处理优化）
        if "Quantile" in self.criterion.__class__.__name__:
            raise NotImplementedError("Quantile损失暂不支持BatchList版本")
        else:
            # 将列表格式转换为批处理格式
            # 从 list[(B, h, w, 3)] x num_views 转换为 list[num_views, h, w, 3] x B
            gt_pts_self_b = torch.unbind(torch.stack(gt_pts_self, dim=1), dim=0)
            pred_pts_self_b = torch.unbind(torch.stack(pred_pts_self, dim=1), dim=0)
            masks_b = torch.unbind(torch.stack(masks, dim=1), dim=0)

            ls_self_b = []  # 批处理损失列表

            # 对每个批次样本计算损失
            for i in range(len(gt_pts_self_b)):
                if depth_only[i]:
                    # 仅深度数据：使用深度尺度平移不变损失
                    ls_self_b.append(
                        self.depth_only_criterion(
                            pred_pts_self_b[i][..., -1],  # 只取深度维度
                            gt_pts_self_b[i][..., -1],    # 只取深度维度
                            masks_b[i],
                        )
                    )
                elif single_view[i] and not is_metric[i]:
                    # 单视图非度量数据：使用尺度不变损失
                    ls_self_b.append(
                        self.single_view_criterion(
                            pred_pts_self_b[i], gt_pts_self_b[i], masks_b[i]
                        )
                    )
                else:
                    # 多视图或度量数据：使用标准损失
                    ls_self_b.append(
                        self.criterion(
                            pred_pts_self_b[i][masks_b[i]],
                            gt_pts_self_b[i][masks_b[i]]
                        )
                    )

            # 重新组织损失数据
            ls_self = self.reorg(ls_self_b, masks_b)

        # 5. 处理天空区域的损失替换
        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_self):
                # 将天空区域的损失替换为固定值
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        # 6. 收集自视角损失的详细信息
        self_name = type(self).__name__
        details = {}
        for i in range(len(ls_self)):
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            # 添加额外的调试信息
            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        # 7. 计算跨视角损失（类似的批处理逻辑）
        if "Quantile" in self.criterion.__class__.__name__:
            raise NotImplementedError("Quantile损失暂不支持BatchList版本")
        else:
            # 排除仅相机参数的样本
            gt_pts_cross_b = torch.unbind(
                torch.stack(gt_pts_cross, dim=1)[~camera_only], dim=0
            )
            pred_pts_cross_b = torch.unbind(
                torch.stack(pred_pts_cross, dim=1)[~camera_only], dim=0
            )
            masks_cross_b = torch.unbind(torch.stack(masks, dim=1)[~camera_only], dim=0)

            ls_cross_b = []

            # 对每个批次样本计算跨视角损失
            for i in range(len(gt_pts_cross_b)):
                if depth_only[~camera_only][i]:
                    # 仅深度数据
                    ls_cross_b.append(
                        self.depth_only_criterion(
                            pred_pts_cross_b[i][..., -1],
                            gt_pts_cross_b[i][..., -1],
                            masks_cross_b[i],
                        )
                    )
                elif single_view[~camera_only][i] and not is_metric[~camera_only][i]:
                    # 单视图非度量数据
                    ls_cross_b.append(
                        self.single_view_criterion(
                            pred_pts_cross_b[i], gt_pts_cross_b[i], masks_cross_b[i]
                        )
                    )
                else:
                    # 标准情况
                    ls_cross_b.append(
                        self.criterion(
                            pred_pts_cross_b[i][masks_cross_b[i]],
                            gt_pts_cross_b[i][masks_cross_b[i]],
                        )
                    )

            # 重新组织跨视角损失
            ls_cross = self.reorg(ls_cross_b, masks_cross_b)

        # 8. 处理跨视角天空区域
        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks_cross = [mask[~camera_only] for mask in masks]
            skys_cross = [sky[~camera_only] for sky in skys]
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        # 9. 收集跨视角损失的详细信息
        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        # 10. 组合所有损失
        ls = ls_self + ls_cross
        masks = masks + masks_cross
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )

        # 11. 计算位姿损失
        pose_masks = pose_masks * gts[i]["img_mask"]  # 应用图像掩码
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class Regr3DPose_ScaleInv(Regr3DPose):
    """
    尺度不变的3D位姿回归损失

    这是Regr3DPose的特殊变体，专门设计用于处理尺度模糊的场景。
    它通过深度归一化来消除绝对尺度的影响，让模型专注于学习相对几何关系。

    核心特点：
    1. 深度归一化：自动消除场景的绝对尺度差异
    2. 形状保持：保持物体的相对形状和比例关系
    3. 尺度适应：适应不同尺度的输入数据

    适用场景：
    - 单目深度估计：存在固有的尺度模糊问题
    - 跨域迁移：源域和目标域的尺度差异很大
    - 形状分析：更关注形状而非绝对大小的任务

    数学原理：
    对于预测和真实的3D点云，分别计算其尺度因子：
    scale_pred = mean(||pred_points - center_pred||)
    scale_gt = mean(||gt_points - center_gt||)

    然后进行归一化：
    pred_normalized = (pred_points - center_pred) / scale_pred
    gt_normalized = (gt_points - center_gt) / scale_gt

    最后计算归一化后的损失。

    在配置文件中的使用：
    test_criterion: Regr3DPose_ScaleInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)

    注意：通常只在测试/评估时使用，用于验证模型的形状预测能力。
    """

    def get_all_pts3d(self, gts, preds):
        """
        获取尺度归一化后的所有3D点数据

        这个方法重写了父类的get_all_pts3d方法，添加了尺度归一化步骤。
        它首先调用父类方法获取基础数据，然后应用尺度归一化。

        参数:
            gts: 真实数据列表
            preds: 预测数据列表

        返回:
            归一化后的3D点数据元组：
            (gt_pts_self, gt_pts_cross, pr_pts_self, pr_pts_cross,
             gt_poses, pr_poses, masks, skys, pose_masks, monitoring)
        """
        # 1. 调用父类方法获取基础的3D点数据
        (
            gt_pts_self,      # 真实自视角3D点
            gt_pts_cross,     # 真实跨视角3D点
            pr_pts_self,      # 预测自视角3D点
            pr_pts_cross,     # 预测跨视角3D点
            gt_poses,         # 真实位姿
            pr_poses,         # 预测位姿
            masks,            # 有效性掩码
            skys,             # 天空掩码
            pose_masks,       # 位姿掩码
            monitoring,       # 监控信息
        ) = super().get_all_pts3d(gts, preds)

        # 2. 计算场景尺度因子
        # 使用工具函数计算点云的中心和尺度
        _, gt_scale_self = get_group_pointcloud_center_scale(gt_pts_self, masks)
        _, pred_scale_self = get_group_pointcloud_center_scale(pr_pts_self, masks)

        _, gt_scale_cross = get_group_pointcloud_center_scale(gt_pts_cross, masks)
        _, pred_scale_cross = get_group_pointcloud_center_scale(pr_pts_cross, masks)

        # 3. 防止预测尺度过于极端
        # 将预测尺度限制在合理范围内，避免数值不稳定
        pred_scale_self = pred_scale_self.clip(min=1e-3, max=1e3)
        pred_scale_cross = pred_scale_cross.clip(min=1e-3, max=1e3)

        # 4. 应用尺度归一化
        if self.gt_scale:
            # 模式1：强制使用GT尺度
            # 将预测点云缩放到GT的尺度
            # 这种模式用于评估：假设我们知道真实尺度，测试形状预测能力
            pr_pts_self = [
                pr_pt_self * gt_scale_self / pred_scale_self
                for pr_pt_self in pr_pts_self
            ]
            pr_pts_cross = [
                pr_pt_cross * gt_scale_cross / pred_scale_cross
                for pr_pt_cross in pr_pts_cross
            ]
        else:
            # 模式2：双向归一化
            # 将GT和预测都归一化到单位尺度
            # 这种模式用于训练：完全消除尺度影响
            gt_pts_self = [gt_pt_self / gt_scale_self for gt_pt_self in gt_pts_self]
            gt_pts_cross = [
                gt_pt_cross / gt_scale_cross for gt_pt_cross in gt_pts_cross
            ]
            pr_pts_self = [pr_pt_self / pred_scale_self for pr_pt_self in pr_pts_self]
            pr_pts_cross = [
                pr_pt_cross / pred_scale_cross for pr_pt_cross in pr_pts_cross
            ]

        # 5. 返回归一化后的数据
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,      # 位姿数据保持不变
            pr_poses,      # 位姿数据保持不变
            masks,         # 掩码保持不变
            skys,          # 天空掩码保持不变
            pose_masks,    # 位姿掩码保持不变
            monitoring,    # 监控信息保持不变
        )

    # 注意：compute_loss方法继承自父类Regr3DPose，无需重写
    # 它会自动使用get_all_pts3d方法返回的归一化数据进行损失计算


"""
=============================================================================
CUT3R损失函数系统完整总结
=============================================================================

经过详细解析，CUT3R的损失函数系统展现出以下设计精髓：

## 1. 层次化架构设计

### 基础层 (Foundation Layer)
- BaseCriterion: 统一的损失函数接口
- LLoss: L范数损失的抽象基类
- L21Loss, MSELoss: 具体的距离度量实现

### 组合层 (Composition Layer)
- MultiLoss: 支持链式组合的损失函数框架
- Criterion: 损失函数的包装和扩展机制
- Sum: 智能的损失聚合函数

### 应用层 (Application Layer)
- RGBLoss: RGB图像重建损失
- DepthScaleShiftInvLoss: 深度尺度平移不变损失
- ScaleInvLoss: 3D点云尺度不变损失
- Regr3DPose: 核心的3D位姿回归损失
- ConfLoss: 置信度自适应学习损失

### 优化层 (Optimization Layer)
- Regr3DPoseBatchList: 批处理优化版本
- Regr3DPose_ScaleInv: 尺度不变特化版本

## 2. 核心设计理念

### 几何一致性 (Geometric Consistency)
- 多视角约束：确保不同视角预测的3D点在几何上一致
- 坐标系变换：正确处理相机坐标系之间的变换关系
- 位姿估计：联合优化3D结构和相机位姿

### 不确定性学习 (Uncertainty Learning)
- 置信度预测：模型学习预测每个像素的可靠性
- 自适应加权：根据置信度动态调整损失权重
- 鲁棒训练：在不确定区域降低损失敏感性

### 尺度处理 (Scale Handling)
- 尺度不变性：处理单目视觉的固有尺度模糊
- 自适应归一化：根据数据特性选择归一化策略
- 跨域适应：支持不同尺度数据集之间的迁移

### 效率优化 (Efficiency Optimization)
- 批处理计算：减少循环开销，提高计算效率
- 内存优化：高效的张量操作，减少内存碎片
- 模块化设计：支持灵活的损失函数组合

## 3. 实际应用指导

### 训练阶段配置
```yaml
train_criterion: ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)
```
- 使用置信度学习提高训练鲁棒性
- 批处理版本提高训练效率
- RGB损失确保图像重建质量

### 测试阶段配置
```yaml
test_criterion: Regr3DPose(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) +
                Regr3DPose_ScaleInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) +
                RGBLoss(L21)
```
- 多重验证确保评估全面性
- 尺度不变版本测试形状预测能力
- 忽略天空区域避免评估偏差

### 医疗数据适配建议
- 保持`norm_mode='?avg_dis'`维护度量特性
- 设置`sky_loss_value=0`忽略无关区域
- 使用适当的尺度缩放因子处理跨域差异

## 4. 技术创新点

1. **置信度加权机制**: 让模型学会"承认"预测的不确定性
2. **多层次尺度处理**: 从像素级到场景级的全方位尺度适应
3. **几何约束融合**: 将2D观测、3D结构和相机位姿统一优化
4. **自适应损失组合**: 根据数据特性动态选择损失函数

这种精心设计的损失函数系统使得CUT3R能够在复杂的3D感知任务中取得卓越性能，
特别是在医疗内窥镜等具有挑战性的专业领域。
"""

"""
总结：CUT3R损失函数系统的设计哲学

1. 模块化设计：
   - BaseCriterion: 提供基础损失接口
   - MultiLoss: 支持损失函数的灵活组合
   - Criterion: 提供损失函数的包装和扩展

2. 多层次损失：
   - RGB损失: 确保图像重建质量
   - 3D几何损失: 确保3D结构的准确性
   - 位姿损失: 确保相机位姿的正确性
   - 置信度损失: 实现自适应的不确定性学习

3. 尺度处理：
   - 深度尺度平移不变: 处理单目深度估计的模糊性
   - 点云尺度不变: 处理3D重建的尺度问题
   - 自适应归一化: 根据数据类型选择合适的归一化策略

4. 几何一致性：
   - 多视角约束: 确保不同视角预测的一致性
   - 坐标系变换: 正确处理不同坐标系之间的变换
   - 掩码处理: 正确处理无效区域和特殊区域（如天空）

这种设计使得CUT3R能够在复杂的3D场景理解任务中取得优异的性能。
"""


# ============================================================================
# Regr3DPose类的完整代码及详细逐行解释
# ============================================================================

class Regr3DPose(Criterion, MultiLoss):
    """
    === 原始代码注释 ===
    Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)

    === 详细解释 ===
    这是CUT3R最核心的损失函数，实现多视角3D点云回归的几何一致性约束。

    核心思想：
    1. 非对称设计：view1作为锚点视角（anchor view），其他视角向它对齐
    2. 双重约束：既要求单视角3D点准确，也要求多视角几何一致
    3. 位姿联合优化：同时优化3D点云预测和相机位姿估计

    数学原理：
    - P1, P2: 世界坐标系下的3D点
    - RT1, RT2: 相机1和相机2的位姿变换矩阵
    - D1, D2: 各自相机坐标系下的3D点
    - loss1: 自视角损失，确保预测的D1准确
    - loss2: 交叉视角损失，确保几何一致性

    关键创新：
    - 支持度量和非度量数据集的统一处理
    - 自适应尺度归一化
    - 天空区域的特殊处理
    - 位姿估计的联合优化
    """

    def __init__(
        self,
        criterion,                  # 基础损失函数（如L21、MSE等）
        norm_mode="?avg_dis",      # 归一化模式，'?'表示只对非度量数据归一化
        gt_scale=False,            # 是否使用真实尺度（测试时为True）
        sky_loss_value=2,          # 天空区域的损失值
        max_metric_scale=False,    # 度量数据的最大尺度限制
    ):
        """
        === 逐行解释 ===
        """
        # 调用父类初始化，设置基础损失函数
        super().__init__(criterion)

        # 解析归一化模式
        if norm_mode.startswith("?"):
            # '?'前缀的含义：对度量尺度数据集不进行归一化
            # 原因：度量数据集（如SCARED）已经有正确的绝对尺度（米为单位）
            # 而非度量数据集（如CO3D）需要归一化来统一尺度
            self.norm_all = False           # 不对所有数据都归一化
            self.norm_mode = norm_mode[1:]  # 去掉'?'前缀，得到实际的归一化模式
        else:
            # 没有'?'前缀：对所有数据集都进行归一化
            self.norm_all = True
            self.norm_mode = norm_mode

        # 是否强制使用真实数据的尺度
        # gt_scale=True: 测试时使用，保持数据的原始尺度
        # gt_scale=False: 训练时使用，允许尺度调整以提高训练稳定性
        self.gt_scale = gt_scale

        # 天空区域的特殊损失值
        # 天空区域通常没有有效的3D结构，给予固定的损失值
        # sky_loss_value=2: 给天空区域一个中等的损失权重
        # sky_loss_value=0: 完全忽略天空区域（测试时常用）
        self.sky_loss_value = sky_loss_value

        # 度量数据集的最大尺度限制
        # 用于过滤距离过远的点，提高训练稳定性
        # 例如：max_metric_scale=100表示忽略距离超过100米的点
        self.max_metric_scale = max_metric_scale

    def get_norm_factor_point_cloud(
        self, pts_self, pts_cross, valids, conf_self, conf_cross, norm_self_only=False
    ):
        """
        === 方法功能 ===
        计算点云归一化因子，用于统一不同场景的尺度

        === 参数说明 ===
        pts_self: 自视角3D点列表 [B, H, W, 3]
        pts_cross: 交叉视角3D点列表 [B, H, W, 3]
        valids: 有效性掩码列表 [B, H, W]
        conf_self: 自视角置信度列表 [B, H, W]
        conf_cross: 交叉视角置信度列表 [B, H, W]
        norm_self_only: 是否只使用自视角进行归一化

        === 逐行解释 ===
        """
        if norm_self_only:
            # 情况1：只使用自视角点云计算归一化因子
            # 适用场景：当交叉视角数据不可靠或缺失时
            norm_factor = normalize_pointcloud_group(
                pts_self,           # 输入：自视角点云列表
                self.norm_mode,     # 归一化模式（如"avg_dis"）
                valids,             # 有效性掩码
                conf_self,          # 置信度权重
                ret_factor_only=True  # 只返回归一化因子，不返回归一化后的点云
            )
        else:
            # 情况2：使用自视角和交叉视角点云联合计算归一化因子
            # 这是默认情况，能够获得更稳定的尺度估计

            # 将自视角和交叉视角点云在最后一个维度上拼接
            # 原理：增加用于归一化的点云数量，提高尺度估计的鲁棒性
            pts = [torch.cat([x, y], dim=2) for x, y in zip(pts_self, pts_cross)]
            # pts[i]: [B, H, W*2, 3] - 每个视角的点云数量翻倍

            # 有效性掩码也需要相应拼接
            # 注意：交叉视角使用与自视角相同的掩码（因为对应同一像素）
            valids = [torch.cat([x, x], dim=2) for x in valids]
            # valids[i]: [B, H, W*2] - 掩码也翻倍

            # 置信度拼接：自视角置信度 + 交叉视角置信度
            confs = [torch.cat([x, y], dim=2) for x, y in zip(conf_self, conf_cross)]
            # confs[i]: [B, H, W*2] - 包含两种置信度信息

            # 使用拼接后的数据计算归一化因子
            norm_factor = normalize_pointcloud_group(
                pts,                # 拼接后的点云
                self.norm_mode,     # 归一化模式
                valids,             # 拼接后的掩码
                confs,              # 拼接后的置信度
                ret_factor_only=True  # 只返回归一化因子
            )
        return norm_factor

    def get_norm_factor_poses(self, gt_trans, pr_trans, not_metric_mask):
        """
        === 方法功能 ===
        计算位姿归一化因子，确保位姿尺度与点云尺度一致

        === 参数说明 ===
        gt_trans: 真实位姿平移部分列表 [B, 3]
        pr_trans: 预测位姿平移部分列表 [B, 3]
        not_metric_mask: 非度量数据掩码 [B] - True表示需要归一化

        === 逐行解释 ===
        """

        # === 步骤1: 计算真实位姿的归一化因子 ===
        if self.norm_mode and not self.gt_scale:
            # 条件：有归一化模式 且 不强制使用GT尺度

            # 将平移向量转换为点云格式以便使用normalize_pointcloud_group
            # 原始形状：[B, 3] -> 目标形状：[B, 1, 1, 3]
            gt_trans = [x[:, None, None, :].clone() for x in gt_trans]
            # 解释：添加两个单例维度，模拟H=1, W=1的"点云"

            # 创建全True的有效性掩码（所有位姿都是有效的）
            valids = [torch.ones_like(x[..., 0], dtype=torch.bool) for x in gt_trans]
            # x[..., 0]的形状：[B, 1, 1] -> valids的形状：[B, 1, 1]

            # 使用点云归一化函数计算位姿的归一化因子
            norm_factor_gt = (
                normalize_pointcloud_group(
                    gt_trans,           # 转换后的"位姿点云"
                    self.norm_mode,     # 归一化模式
                    valids,             # 有效性掩码
                    ret_factor_only=True,  # 只返回归一化因子
                )
                .squeeze(-1)            # 去掉最后一个维度：[B, 1, 1] -> [B, 1]
                .squeeze(-1)            # 再去掉一个维度：[B, 1] -> [B]
            )
        else:
            # 情况：不需要归一化（度量数据集或强制使用GT尺度）
            # 创建全1的归一化因子（即不进行任何缩放）
            norm_factor_gt = torch.ones(
                len(gt_trans),              # 批次大小
                dtype=gt_trans[0].dtype,    # 数据类型与输入一致
                device=gt_trans[0].device   # 设备与输入一致
            )

        # === 步骤2: 计算预测位姿的归一化因子 ===
        # 初始化：预测位姿的归一化因子与真实位姿相同
        norm_factor_pr = norm_factor_gt.clone()

        # 对非度量数据的预测位姿进行特殊处理
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            # 条件：有归一化模式 且 存在非度量数据 且 不强制使用GT尺度

            # 提取非度量数据的预测位姿
            pr_trans_not_metric = [
                x[not_metric_mask][:, None, None, :].clone() for x in pr_trans
            ]
            # 只处理标记为非度量的样本，并转换为点云格式

            # 为非度量数据创建有效性掩码
            valids = [
                torch.ones_like(x[..., 0], dtype=torch.bool)
                for x in pr_trans_not_metric
            ]

            # 单独计算非度量数据的预测位姿归一化因子
            norm_factor_pr_not_metric = (
                normalize_pointcloud_group(
                    pr_trans_not_metric,    # 非度量数据的预测位姿
                    self.norm_mode,         # 归一化模式
                    valids,                 # 有效性掩码
                    ret_factor_only=True,   # 只返回归一化因子
                )
                .squeeze(-1)
                .squeeze(-1)
            )

            # 将非度量数据的归一化因子更新到对应位置
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        # 返回真实位姿和预测位姿的归一化因子
        return norm_factor_gt, norm_factor_pr

    def get_all_pts3d(
        self,
        gts,                        # 真实数据列表
        preds,                      # 预测数据列表
        dist_clip=None,             # 距离裁剪阈值
        norm_self_only=False,       # 是否只使用自视角归一化
        norm_pose_separately=False, # 是否单独归一化位姿
        eps=1e-3,                   # 数值稳定性参数
        camera1=None,               # 参考相机（默认使用第一个视角）
    ):
        """
        === 方法功能 ===
        这是损失计算的核心方法，负责：
        1. 统一坐标系：将所有3D点转换到参考坐标系
        2. 数据预处理：距离裁剪、掩码处理、置信度提取
        3. 尺度归一化：统一不同场景的尺度
        4. 位姿处理：计算相对位姿并进行归一化
        5. 特殊情况处理：处理仅相机监督等特殊情况

        === 逐行解释 ===
        """

        # === 步骤1: 坐标系统一 ===
        # everything is normalized w.r.t. camera of view1
        # 选择参考相机的逆变换矩阵（通常是第一个视角）
        in_camera1 = inv(gts[0]["camera_pose"]) if camera1 is None else inv(camera1)
        # 解释：inv()计算相机位姿的逆矩阵，用于将世界坐标转换到相机坐标

        # 将真实3D点转换到各自的相机坐标系（self view）
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        # 解释：geotrf()执行几何变换，将世界坐标系的3D点转换到各自相机坐标系
        # gt["pts3d"]: 世界坐标系下的3D点 [B, H, W, 3]
        # inv(gt["camera_pose"]): 世界到相机的变换矩阵
        # 结果：每个视角在自己坐标系下的3D点

        # 将真实3D点转换到参考相机坐标系（cross view）
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        # 解释：将所有视角的3D点都转换到第一个视角的坐标系
        # 这样可以在统一坐标系下比较不同视角的预测结果

        # 提取有效性掩码
        valids = [gt["valid_mask"].clone() for gt in gts]
        # 解释：valid_mask标记哪些像素有有效的3D信息
        # clone()确保不会修改原始数据

        # 检查是否有仅相机监督的数据
        camera_only = gts[0]["camera_only"]
        # 解释：某些数据可能只有相机内参监督，没有3D点监督

        # === 步骤2: 距离裁剪（可选） ===
        if dist_clip is not None:
            # points that are too far-away == invalid
            # 计算每个3D点到参考相机的距离
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            # gt_pt.norm(dim=-1): 计算每个点的L2范数（距离）
            # 形状：[B, H, W, 3] -> [B, H, W]

            # 过滤距离过远的点
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]
            # 解释：距离超过dist_clip的点被标记为无效
            # 目的：提高训练稳定性，避免极远点的影响

        # === 步骤3: 提取预测数据 ===
        # 提取预测的自视角3D点
        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        # 解释：模型预测的在各自相机坐标系下的3D点

        # 提取预测的交叉视角3D点
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        # 解释：模型预测的变换到参考坐标系的3D点

        # 提取置信度（使用对数空间提高数值稳定性）
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        # 解释：
        # - pred["conf_self"]: 自视角置信度 [B, H, W]
        # - torch.log(): 转换到对数空间，避免置信度过小导致的数值问题
        # - .detach(): 不计算置信度的梯度（置信度用于加权，不直接优化）
        # - .clip(eps): 限制最小值，防止log(0)的情况

        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]
        # 解释：交叉视角置信度，处理方式与自视角相同

        # === 步骤4: 确定度量数据掩码 ===
        if not self.norm_all:
            # 情况：不对所有数据都进行归一化（有'?'前缀）

            if self.max_metric_scale:
                # 如果设置了最大度量尺度限制
                B = valids[0].shape[0]  # 批次大小

                # 计算每个样本中所有有效点的最大距离
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                # 解释：
                # - torch.linalg.norm(gt_pt_cross, dim=-1): 计算每个点的距离
                # - torch.where(valid, distance, 0): 只考虑有效点的距离
                # - .view(B, -1): 重塑为[B, H*W]便于计算最大值

                # 更新度量数据掩码：距离不能超过最大限制
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
                # 解释：
                # - d.max(dim=-1).values: 每个样本的最大距离 [B]
                # - < self.max_metric_scale: 检查是否超过限制
                # - &: 与原始度量掩码取交集

            # 非度量数据掩码（需要归一化的数据）
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            # 情况：对所有数据都进行归一化
            # 将所有数据都视为非度量数据
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # === 步骤5: 3D点云归一化 ===
        # normalize 3d points
        # compute the scale using only the self view point maps

        # 计算真实3D点的归一化因子
        if self.norm_mode and not self.gt_scale:
            # 条件：有归一化模式 且 不强制使用GT尺度
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self,            # 真实自视角3D点
                gt_pts_cross,           # 真实交叉视角3D点
                valids,                 # 有效性掩码
                conf_self,              # 自视角置信度
                conf_cross,             # 交叉视角置信度
                norm_self_only=norm_self_only,  # 是否只使用自视角
            )
        else:
            # 情况：不需要归一化
            # 创建全1的归一化因子（形状与预测数据匹配）
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )
            # 解释：[:, :1, :1, :1]创建形状为[B, 1, 1, 1]的张量

        # 预测3D点的归一化因子初始化
        norm_factor_pr = norm_factor_gt.clone()

        # 对非度量数据的预测3D点单独计算归一化因子
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            # 条件：有归一化模式 且 存在非度量数据 且 不强制使用GT尺度

            norm_factor_pr_not_metric = self.get_norm_factor_point_cloud(
                # 只提取非度量数据的预测点云
                [pr_pt_self[not_metric_mask] for pr_pt_self in pr_pts_self],
                [pr_pt_cross[not_metric_mask] for pr_pt_cross in pr_pts_cross],
                [valid[not_metric_mask] for valid in valids],
                [conf[not_metric_mask] for conf in conf_self],
                [conf[not_metric_mask] for conf in conf_cross],
                norm_self_only=norm_self_only,
            )
            # 解释：[tensor[mask] for tensor in tensor_list]
            # 对列表中每个张量应用掩码，只保留非度量数据的部分

            # 将非度量数据的归一化因子更新到对应位置
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        # 数值稳定性处理：防止归一化因子过小
        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        # 应用归一化因子到所有3D点
        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]
        # 解释：除法操作会广播，norm_factor形状[B,1,1,1]会扩展到[B,H,W,3]

        # === 步骤6: 位姿处理 ===
        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion

        # 计算真实位姿（相对于参考相机）
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        # 解释：
        # - in_camera1 @ gt["camera_pose"]: 计算相对位姿变换
        # - camera_to_pose_encoding(): 将4x4变换矩阵转换为(平移3D, 四元数4D)格式
        # - 结果：每个位姿表示为(Bx3, Bx4)的元组

        # 提取预测位姿
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        # 解释：预测位姿已经是相对于参考相机的格式

        # 位姿归一化因子（从点云归一化因子派生）
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)
        # 解释：从[B,1,1,1]压缩到[B,1]，适配位姿的形状

        # === 步骤7: 位姿归一化的特殊处理 ===
        if norm_pose_separately:
            # 情况1：单独对位姿进行归一化
            gt_trans = [gt[:, :3] for gt in gt_poses]    # 提取平移部分
            pr_trans = [pr[:, :3] for pr in pr_poses]    # 提取平移部分
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, not_metric_mask
            )
        elif any(camera_only):
            # 情况2：存在仅相机监督的数据
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(gt_trans, pr_trans, not_metric_mask)
            )
            # 根据camera_only掩码选择使用哪种归一化因子
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )
            # 解释：torch.where根据camera_only掩码选择归一化因子
            # camera_only=True: 使用专门的位姿归一化因子
            # camera_only=False: 使用从点云派生的归一化因子

        # === 步骤8: 应用位姿归一化 ===
        # 对位姿的平移部分应用归一化，旋转部分保持不变
        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        # 解释：
        # - gt[:, :3]: 平移部分 [B, 3]
        # - gt[:, 3:]: 四元数部分 [B, 4]
        # - 只对平移部分进行归一化，旋转不受尺度影响

        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]

        # 创建位姿有效性掩码
        pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
            pose_norm_factor_pr.squeeze() > eps
        )
        # 解释：只有当真实和预测的归一化因子都有效时，位姿损失才有效

        # === 步骤9: 仅相机监督的特殊处理 ===
        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            # 当只有相机内参监督时，将3D点转换为归一化坐标

            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],  # 广播掩码到[B,H,W,3]
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),  # 透视除法+裁剪
                    gt,  # 保持原始3D坐标
                )
                for gt in gt_pts_self
            ]
            # 解释：
            # - gt[..., -1:]: Z坐标（深度） [B,H,W,1]
            # - gt / gt[..., -1:]: 透视除法，得到归一化坐标 [X/Z, Y/Z, 1]
            # - .clip(1e-6): 防止除零
            # - .clip(-2, 2): 限制归一化坐标范围，提高数值稳定性

            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision
            # 注释：仅相机监督时不添加交叉视角损失

        # === 步骤10: 天空区域处理 ===
        # 天空区域通常没有有效的3D结构，需要特殊处理
        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        # 解释：
        # - gt["sky_mask"]: 天空区域掩码
        # - ~valid: 无效区域掩码
        # - &: 取交集，得到天空且无效的区域
        # 这些区域将在损失计算中给予特殊的损失值

        # === 返回所有处理后的数据 ===
        return (
            gt_pts_self,    # 归一化后的真实自视角3D点
            gt_pts_cross,   # 归一化后的真实交叉视角3D点
            pr_pts_self,    # 归一化后的预测自视角3D点
            pr_pts_cross,   # 归一化后的预测交叉视角3D点
            gt_poses,       # 归一化后的真实位姿
            pr_poses,       # 归一化后的预测位姿
            valids,         # 有效性掩码
            skys,           # 天空区域掩码
            pose_masks,     # 位姿有效性掩码
            {},             # 额外的监控信息（此版本为空）
        )

    def compute_pose_loss(self, gt_poses, pred_poses, masks=None):
        """
        === 原始注释 ===
        gt_pose: list of (Bx3, Bx4)
        pred_pose: list of (Bx3, Bx4)
        masks: None, or B

        === 方法功能 ===
        计算相机位姿损失，包括平移和旋转两部分

        === 逐行解释 ===
        """
        # 提取并堆叠真实位姿的平移部分
        gt_trans = torch.stack([gt[0] for gt in gt_poses], dim=1)  # BxNx3
        # 解释：
        # - gt_poses: [(B,3), (B,4)] 的列表，每个元素是一个视角的位姿
        # - gt[0]: 提取平移部分 (B,3)
        # - torch.stack(..., dim=1): 在第1维堆叠，得到 (B,N,3)
        # - N是视角数量

        # 提取并堆叠真实位姿的四元数部分
        gt_quats = torch.stack([gt[1] for gt in gt_poses], dim=1)  # BxNx4
        # 解释：gt[1]是四元数部分 (B,4)，堆叠后得到 (B,N,4)

        # 提取并堆叠预测位姿的平移部分
        pred_trans = torch.stack([pr[0] for pr in pred_poses], dim=1)  # BxNx3

        # 提取并堆叠预测位姿的四元数部分
        pred_quats = torch.stack([pr[1] for pr in pred_poses], dim=1)  # BxNx4

        # 根据是否有掩码计算损失
        if masks == None:
            # 情况1：没有掩码，计算所有样本的损失
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1).mean()
                + torch.norm(pred_quats - gt_quats, dim=-1).mean()
            )
            # 解释：
            # - torch.norm(..., dim=-1): 计算最后一维的L2范数
            # - pred_trans - gt_trans: 形状(B,N,3)，计算平移误差
            # - norm后形状变为(B,N)，表示每个样本每个视角的平移误差
            # - .mean(): 对所有样本和视角求平均
        else:
            # 情况2：有掩码，只计算有效样本的损失

            # 增加一个健壮的检查，同时处理列表和张量类型的 masks
            is_empty = False
            if isinstance(masks, torch.Tensor):
                # 张量类型的掩码
                if not torch.any(masks):
                    is_empty = True
            elif not any(masks):
                # 列表类型的掩码
                is_empty = True

            if is_empty:
                # 如果所有掩码都是False，返回零损失
                return torch.tensor(0.0, device=pred_trans.device)

            # 计算有效样本的位姿损失
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1)[masks].mean()
                + torch.norm(pred_quats - gt_quats, dim=-1)[masks].mean()
            )
            # 解释：
            # - [masks]: 应用掩码，只选择有效的样本
            # - masks的形状应该是(B,)，会广播到(B,N)

        return pose_loss

    def compute_loss(self, gts, preds, **kw):
        """
        === 方法功能 ===
        这是Regr3DPose类的主要接口，计算完整的多视角3D损失

        处理流程：
        1. 数据预处理：调用get_all_pts3d获取归一化的数据
        2. 自视角损失：计算每个视角在自己坐标系下的3D点损失
        3. 交叉视角损失：计算多视角几何一致性损失
        4. 位姿损失：计算相机位姿估计损失
        5. 特殊处理：天空区域、置信度加权等

        === 逐行解释 ===
        """

        # === 步骤1: 数据预处理 ===
        # 调用get_all_pts3d获取所有预处理后的数据
        (
            gt_pts_self,    # 真实自视角3D点
            gt_pts_cross,   # 真实交叉视角3D点
            pred_pts_self,  # 预测自视角3D点
            pred_pts_cross, # 预测交叉视角3D点
            gt_poses,       # 真实位姿
            pr_poses,       # 预测位姿
            masks,          # 有效性掩码
            skys,           # 天空区域掩码
            pose_masks,     # 位姿有效性掩码
            monitoring,     # 监控信息
        ) = self.get_all_pts3d(gts, preds, **kw)

        # === 步骤2: 天空区域处理 ===
        if self.sky_loss_value > 0:
            # 确保基础损失函数使用"none"模式（返回每个像素的损失）
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"

            # 将天空区域添加到有效掩码中
            masks = [mask | sky for mask, sky in zip(masks, skys)]
            # 解释：mask | sky 表示有效区域或天空区域都会计算损失
            # 天空区域将在后续步骤中给予特殊的损失值

        # === 步骤3: 自视角损失计算 ===
        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            # 特殊情况：使用分位数损失
            # masks are overwritten taking into account self view losses
            ls_self, masks = self.criterion(
                pred_pts_self, gt_pts_self, masks, gts[0]["quantile"]
            )
        else:
            # 标准情况：使用常规损失函数
            ls_self = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(pred_pts_self, gt_pts_self, masks)
            ]
            # 解释：
            # - pred_pt[mask]: 只选择有效像素的预测3D点
            # - gt_pt[mask]: 只选择有效像素的真实3D点
            # - self.criterion: 基础损失函数（如L21）
            # - 结果：每个视角的自视角损失

        # === 步骤4: 天空区域的特殊损失值 ===
        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"

            for i, l in enumerate(ls_self):
                # 对天空区域应用特殊的损失值
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)
                # 解释：
                # - skys[i][masks[i]]: 在有效掩码范围内的天空区域
                # - self.sky_loss_value: 天空区域的固定损失值（如2.0）
                # - torch.where: 天空区域使用固定值，其他区域使用计算的损失

        # === 步骤5: 收集详细信息 ===
        self_name = type(self).__name__  # 获取类名用于日志

        details = {}
        for i in range(len(ls_self)):
            # 记录每个视角的自视角损失
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())

            # 保存可视化数据
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            # 保存CUT3R特有的掩码信息
            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()
                # 解释：CUT3R使用双路编码器，需要记录图像和射线映射的掩码

            # 保存描述符信息（如果有的话）
            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        # === 步骤6: 交叉视角损失计算 ===
        # cross view loss and details

        # 过滤掉仅相机监督的样本（它们不参与交叉视角损失）
        camera_only = gts[0]["camera_only"]
        pred_pts_cross = [pred_pts[~camera_only] for pred_pts in pred_pts_cross]
        gt_pts_cross = [gt_pts[~camera_only] for gt_pts in gt_pts_cross]
        masks_cross = [mask[~camera_only] for mask in masks]
        skys_cross = [sky[~camera_only] for sky in skys]
        # 解释：
        # - ~camera_only: 取反，选择非仅相机监督的样本
        # - 仅相机监督的样本只有内参信息，无法计算交叉视角几何一致性

        # 计算交叉视角损失
        if "Quantile" in self.criterion.__class__.__name__:
            # 分位数损失的特殊处理
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            ls_cross, _ = self.criterion(
                pred_pts_cross, gt_pts_cross, masks_cross, None
            )
        else:
            # 标准交叉视角损失计算
            ls_cross = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(
                    pred_pts_cross, gt_pts_cross, masks_cross
                )
            ]
            # 解释：与自视角损失类似，但使用交叉视角的数据

        # === 步骤7: 交叉视角的天空区域处理 ===
        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"

            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        # === 步骤8: 记录交叉视角损失详情 ===
        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            # 解释：
            # - ls_cross[i].numel() > 0: 检查是否有有效元素
            # - 如果没有有效元素（如全是仅相机监督），损失为0

            details[f"conf_{i+1}"] = preds[i]["conf"].detach()
            # 保存交叉视角置信度

        # === 步骤9: 组合所有损失 ===
        # 合并自视角和交叉视角损失
        ls = ls_self + ls_cross
        masks = masks + masks_cross

        # 记录损失类型信息
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        # 解释：标记每个损失是自视角(True)还是交叉视角(False)

        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )
        # 解释：记录每个损失对应的图像ID

        # === 步骤10: 计算位姿损失 ===
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        # === 返回最终结果 ===
        # Sum类将损失和掩码组合，实现加权平均
        return Sum(*list(zip(ls, masks))), (details | monitoring)
        # 解释：
        # - Sum(*list(zip(ls, masks))): 创建加权损失对象
        # - details | monitoring: 合并详细信息和监控信息
        # - 返回(损失对象, 详细信息字典)的元组


# ============================================================================
# 总结：Regr3DPose类的核心价值
# ============================================================================
"""
Regr3DPose类是CUT3R最核心的损失函数，它实现了以下关键功能：

1. 多视角几何一致性：
   - 自视角损失：确保每个视角的3D预测准确
   - 交叉视角损失：确保不同视角的几何一致性
   - 位姿损失：联合优化相机位姿估计

2. 自适应尺度处理：
   - 度量数据集：保持真实尺度（如SCARED的米单位）
   - 非度量数据集：自动归一化尺度（如CO3D）
   - 混合数据集：智能区分处理

3. 鲁棒性设计：
   - 天空区域特殊处理：避免无效3D结构的干扰
   - 距离裁剪：过滤过远点提高稳定性
   - 数值稳定性：防止除零和梯度爆炸

4. 灵活的监督模式：
   - 完整3D监督：使用所有损失项
   - 仅相机监督：只使用内参相关损失
   - 置信度加权：自适应调整不同区域的损失权重

这种设计使得CUT3R能够在各种复杂场景下实现准确的连续3D感知。
"""
