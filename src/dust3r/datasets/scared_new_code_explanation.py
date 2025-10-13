"""
SCARED 数据集读取代码详细解释
============================

本文件对 /hy-tmp/CUT3R/src/dust3r/datasets/scared_new.py 进行逐行详细解释
SCARED (Stereo Correspondence And REconstruction of Endoscopic Data) 是一个专门用于内窥镜手术场景的立体视觉数据集

作者：AI Assistant  
日期：2024
"""

# ==================== 导入模块部分 ====================

import os.path as osp
# os.path 模块用于处理文件路径，osp 是常用的别名
# 提供跨平台的路径操作功能，如 join(), dirname(), exists() 等

import cv2
# OpenCV 计算机视觉库，用于图像处理和计算机视觉任务
# 在这里主要用于图像的读取、处理和变换操作

import numpy as np
# NumPy 数组计算库，用于高效的数值计算
# 在深度学习和计算机视觉中广泛用于数组操作

import itertools
# Python 内置模块，提供迭代器工具
# 虽然在这个文件中导入了但实际没有使用

import os
# 操作系统接口模块，用于与操作系统交互
# 主要用于文件和目录操作，如 listdir(), join() 等

import sys
# 系统相关的参数和函数模块
# 在这里用于修改 Python 路径以导入自定义模块

# ==================== 路径设置部分 ====================

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
# 将当前文件的祖父目录添加到 Python 路径中
# 这样可以导入项目根目录下的模块
# __file__ 是当前文件的路径
# osp.dirname(__file__) 获取当前文件所在目录
# "..", ".." 表示向上两级目录

from tqdm import tqdm
# tqdm 是一个进度条库，用于显示循环和迭代的进度
# 在加载大量数据时提供可视化的进度反馈

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
# 导入基础的多视角数据集类
# SCARED_Multi 类将继承这个基类
# BaseMultiViewDataset 提供了多视角数据集的通用功能

from dust3r.utils.image import imread_cv2
# 导入自定义的图像读取函数
# imread_cv2 是一个封装的图像读取函数，使用 OpenCV 后端

# ==================== SCARED 数据集类定义 ====================

class SCARED_Multi(BaseMultiViewDataset):
    """
    SCARED 多视角数据集类
    
    SCARED (Stereo Correspondence And REconstruction of Endoscopic Data) 是一个
    专门为内窥镜手术场景设计的立体视觉数据集，包含：
    - 多视角RGB图像（来自立体内窥镜相机）
    - 深度图（通过立体匹配或激光扫描生成）
    - 相机参数（内参和外参）
    - 手术器械和组织的3D重建数据
    
    该类负责加载和处理 SCARED 数据集的数据，特别适用于医疗机器人和手术导航
    """
    
    def __init__(self, *args, ROOT, **kwargs):
        """
        初始化 SCARED 数据集
        
        参数:
            *args: 位置参数，传递给父类
            ROOT: 数据集根目录路径
            **kwargs: 关键字参数，传递给父类
        """
        
        self.ROOT = ROOT
        # 设置数据集根目录路径
        # ROOT 应该指向包含 train/test/val 子目录的主目录
        
        self.video = True
        # 标记这是视频数据集（时序数据）
        # 表示图像之间存在时间关系，可以用于视频相关的任务
        # 在手术场景中，这对于跟踪器械运动和组织变形很重要
        
        self.is_metric = True
        # 标记深度信息是度量深度（真实尺度）
        # True 表示深度值有真实的物理意义（以米为单位）
        # 这对于手术场景的精确测量至关重要
        
        self.max_interval = 4
        # 设置最大时间间隔为4帧
        # 在选择图像序列时，相邻图像的最大时间间隔不超过4帧
        # 这确保手术器械和组织的运动在视觉上保持连续性
        
        super().__init__(*args, **kwargs)
        # 调用父类的初始化方法
        # 父类 BaseMultiViewDataset 会处理通用的多视角数据集初始化
        
        assert self.split in ["train", "test", "val"]
        # 断言检查数据集分割必须是训练集、测试集或验证集之一
        # self.split 应该在父类初始化时设置
        
        # ==================== SCARED 数据集场景列表 ====================
        self.scenes_to_use = [
            # Dataset 1 - 第一组手术场景
            "dataset1_keyframe1",    # 数据集1的关键帧1
            "dataset1_keyframe2",    # 数据集1的关键帧2
            "dataset1_keyframe3",    # 数据集1的关键帧3
            
            # Dataset 2 - 第二组手术场景
            "dataset2_keyframe1",    # 数据集2的关键帧1
            "dataset2_keyframe2",    # 数据集2的关键帧2
            "dataset2_keyframe3",    # 数据集2的关键帧3
            "dataset2_keyframe4",    # 数据集2的关键帧4
            
            # Dataset 3 - 第三组手术场景
            "dataset3_keyframe1",    # 数据集3的关键帧1
            "dataset3_keyframe2",    # 数据集3的关键帧2
            "dataset3_keyframe3",    # 数据集3的关键帧3
            "dataset3_keyframe4",    # 数据集3的关键帧4
            
            # Dataset 6 - 第六组手术场景
            "dataset6_keyframe1",    # 数据集6的关键帧1
            "dataset6_keyframe2",    # 数据集6的关键帧2
            "dataset6_keyframe3",    # 数据集6的关键帧3
            "dataset6_keyframe4",    # 数据集6的关键帧4
            
            # Dataset 7 - 第七组手术场景
            "dataset7_keyframe1",    # 数据集7的关键帧1
            "dataset7_keyframe2",    # 数据集7的关键帧2
            "dataset7_keyframe3",    # 数据集7的关键帧3
            "dataset7_keyframe4",    # 数据集7的关键帧4
        ]
        # SCARED 数据集的命名规则：datasetX_keyframeY
        # - X 表示不同的手术场景或病例
        # - Y 表示该场景中的关键帧编号
        # 关键帧是从连续视频中选择的代表性帧，包含重要的手术信息
        # 注意：这里没有 dataset4 和 dataset5，可能是因为质量问题或其他原因被排除
        
        self.loaded_data = self._load_data(self.split)
        # 调用数据加载方法，根据当前的数据集分割加载相应的数据
        # 返回值存储在 self.loaded_data 中（虽然这里没有使用返回值）

    def _load_data(self, split):
        """
        加载指定分割的数据
        
        这个方法扫描数据集目录，收集所有可用的图像和场景信息
        
        参数:
            split: 数据集分割名称 ("train", "test", "val")
        """
        
        root = os.path.join(self.ROOT, split)
        # 构建当前分割的根目录路径
        # 例如：/path/to/scared/train
        
        self.scenes = []
        # 初始化场景列表，存储实际使用的场景名称

        # ==================== 数据统计变量初始化 ====================
        offset = 0
        # 图像索引的偏移量，用于给每个图像分配全局唯一的ID
        
        scenes = []
        # 场景名称列表
        
        sceneids = []
        # 每个图像对应的场景ID列表
        
        scene_img_list = []
        # 每个场景包含的图像ID列表
        
        images = []
        # 所有图像的基础名称列表（不包含扩展名）
        
        start_img_ids = []
        # 可以作为起始图像的ID列表
        # 这些是可以开始多视角序列的图像

        j = 0  # 场景计数器

        # ==================== 遍历所有场景目录 ====================
        for scene in tqdm(os.listdir(root)):
            # 使用 tqdm 显示进度条，遍历根目录下的所有场景文件夹
            
            if scene not in self.scenes_to_use:
                continue
            # 如果当前场景不在预定义的使用列表中，跳过
            # 这样可以过滤掉不需要的场景或有问题的数据
            
            scene_dir = osp.join(root, scene)
            # 构建当前场景的完整路径
            
            rgb_dir = osp.join(scene_dir, "rgb")
            # 构建RGB图像目录路径
            # SCARED 数据集的结构：scene/rgb/image.jpg
            
            # ==================== 获取图像文件列表 ====================
            basenames = sorted([f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
            # 获取所有 .jpg 文件的基础名称（去掉扩展名）
            # sorted() 确保文件按顺序排列，这对视频序列很重要
            # f[:-4] 去掉文件名的最后4个字符（.jpg）
            # 在手术场景中，图像的时序顺序对于理解手术过程很重要
            
            num_imgs = len(basenames)
            # 当前场景的图像数量
            
            img_ids = list(np.arange(num_imgs) + offset)
            # 为当前场景的所有图像分配全局唯一的ID
            # offset 确保不同场景的图像ID不重复
            
            # ==================== 计算可用的起始图像 ====================
            cut_off = (self.num_views if not self.allow_repeat else max(self.num_views // 3, 3))
            # 计算截止值：
            # 如果不允许重复视图，截止值等于所需视图数
            # 如果允许重复视图，截止值是视图数的1/3（最少3个）
            # 这确保有足够的图像来构成一个完整的多视角序列
            
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            # 计算可以作为起始点的图像ID
            # 从第一个图像到倒数第cut_off个图像都可以作为起始点
            # 这确保从起始点开始能够采样到足够的后续图像

            # ==================== 场景有效性检查 ====================
            if num_imgs < cut_off:
                print(f"Skipping {scene} (too few images: {num_imgs})")
                continue
            # 如果当前场景的图像数量少于截止值，跳过这个场景
            # 这确保每个场景都有足够的图像来构成多视角序列
            # 在手术数据中，确保足够的图像数量对于重建手术场景很重要

            # ==================== 更新全局数据结构 ====================
            start_img_ids.extend(start_img_ids_)
            # 将当前场景的起始图像ID添加到全局列表
            
            sceneids.extend([j] * num_imgs)
            # 为当前场景的所有图像分配场景ID
            # [j] * num_imgs 创建一个长度为 num_imgs 的列表，所有元素都是 j
            
            images.extend(basenames)
            # 将当前场景的所有图像基础名称添加到全局列表
            
            scenes.append(scene)
            # 将场景名称添加到场景列表
            
            scene_img_list.append(img_ids)
            # 将当前场景的图像ID列表添加到场景图像列表

            # ==================== 更新计数器和偏移量 ====================
            offset += num_imgs
            # 更新偏移量，为下一个场景的图像ID做准备
            
            j += 1
            # 场景计数器加1

        # ==================== 保存加载的数据 ====================
        self.scenes = scenes
        # 保存实际使用的场景名称列表
        
        self.sceneids = sceneids
        # 保存每个图像对应的场景ID
        
        self.images = images
        # 保存所有图像的基础名称
        
        self.start_img_ids = start_img_ids
        # 保存所有可能的起始图像ID
        
        self.scene_img_list = scene_img_list
        # 保存每个场景的图像ID列表

    def __len__(self):
        """
        返回数据集的长度
        
        数据集的长度等于可能的起始点数量
        每个起始点可以构成一个多视角序列样本
        """
        return len(self.start_img_ids)

    def get_image_num(self):
        """
        返回数据集中图像的总数量
        
        这包括所有场景中的所有图像
        """
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        """
        获取指定索引的多视角数据
        
        这是数据集的核心方法，负责加载和处理多视角图像数据
        特别针对手术场景进行了优化
        
        参数:
            idx: 数据样本的索引
            resolution: 目标图像分辨率
            rng: 随机数生成器
            num_views: 需要的视图数量
            
        返回:
            views: 包含多个视图数据的列表
        """
        
        start_id = self.start_img_ids[idx]
        # 获取当前样本的起始图像ID
        
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        # 获取起始图像所属场景的所有图像ID
        # self.sceneids[start_id] 获取起始图像的场景ID
        # self.scene_img_list[场景ID] 获取该场景的所有图像ID
        
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,           # 需要的视图数量
            start_id,           # 起始图像ID
            all_image_ids,      # 场景中所有可用的图像ID
            rng,                # 随机数生成器
            max_interval=self.max_interval,  # 最大时间间隔
            video_prob=1.0,     # 视频序列概率（1.0表示总是选择时序连续的图像）
            fix_interval_prob=1.0,  # 固定间隔概率
        )
        # 这个方法（来自父类）选择一个图像序列
        # pos: 选中图像在 all_image_ids 中的位置索引
        # ordered_video: 是否是有序的视频序列
        # 在手术场景中，保持时序连续性对于理解手术过程很重要
        
        image_idxs = np.array(all_image_ids)[pos]
        # 根据位置索引获取实际的图像ID
        # 将 all_image_ids 转换为 numpy 数组，然后用 pos 索引

        views = []
        # 初始化视图列表
        
        # ==================== 加载每个视图的数据 ====================
        for v, view_idx in enumerate(image_idxs):
            # 遍历选中的每个图像ID
            # v: 视图索引（0, 1, 2, ...）
            # view_idx: 图像的全局ID
            
            scene_id = self.sceneids[view_idx]
            # 获取当前图像所属的场景ID
            
            # ==================== 构建文件路径 ====================
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            # 构建场景目录路径
            # 例如：/path/to/scared/train/dataset1_keyframe1
            
            rgb_dir = osp.join(scene_dir, "rgb")
            # RGB图像目录
            
            depth_dir = osp.join(scene_dir, "depth")
            # 深度图目录
            
            cam_dir = osp.join(scene_dir, "cam")
            # 相机参数目录

            basename = self.images[view_idx]
            # 获取当前图像的基础名称

            # ==================== 加载RGB图像 ====================
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            # 使用自定义的 imread_cv2 函数加载RGB图像
            # 文件路径：rgb_dir/basename.jpg
            # 手术场景中的RGB图像通常包含手术器械、组织和血液等复杂内容
            
            # ==================== 加载和处理深度图 ====================
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            # 加载对应的深度图，深度数据以 .npy 格式存储
            
            depthmap[~np.isfinite(depthmap)] = 0  # 处理无效值
            # ~np.isfinite() 找到所有非有限值（NaN、无穷大等）
            # 在手术场景中，反光表面和血液可能导致深度估计失败
            
            # SCARED深度单位是米，保持原始单位（is_metric=True）
            depthmap[depthmap > 1.0] = 0.0  # 过滤超过1米的深度值（适合手术场景）
            # 手术场景的深度范围通常在几厘米到几十厘米之间
            # 超过1米的深度值可能是错误的测量，特别是在腹腔镜手术中
            # 这个阈值专门针对内窥镜手术场景进行了优化

            # ==================== 加载相机参数 ====================
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            # 加载相机参数文件，.npz 是 numpy 的压缩格式
            
            camera_pose = cam["pose"]
            # 相机位姿（外参）：4x4变换矩阵，包含旋转和平移
            # 在手术场景中，这表示内窥镜相机的位置和方向
            
            intrinsics = cam["intrinsics"]
            # 相机内参：3x3矩阵，包含焦距、主点等参数
            # 内窥镜相机通常有特殊的光学特性，需要精确的内参校正
            
            # ==================== 图像预处理 ====================
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )
            # 根据需要裁剪和缩放图像
            # 这个方法会同时调整RGB图像、深度图和相机内参
            # 确保所有数据保持一致性
            # 在手术场景中，图像预处理要特别小心保持深度信息的准确性

            # ==================== 生成掩码 ====================
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.9, 0.05, 0.05]
            )
            # 生成图像掩码和光线掩码
            # p=[0.9, 0.05, 0.05] 可能表示不同掩码类型的概率分布
            # img_mask: 图像级别的掩码
            # ray_mask: 光线级别的掩码（用于射线投射等算法）
            # 在手术场景中，掩码可以用于忽略手术器械或血液区域

            # ==================== 构建视图数据字典 ====================
            views.append(
                dict(
                    img=rgb_image,                                    # RGB图像数据
                    depthmap=depthmap.astype(np.float32),            # 深度图（转换为float32）
                    camera_pose=camera_pose.astype(np.float32),      # 相机位姿（外参）
                    camera_intrinsics=intrinsics.astype(np.float32), # 相机内参
                    dataset="SCARED",                                # 数据集名称标识
                    label=self.scenes[scene_id] + "_" + basename,    # 数据标签（场景名_图像名）
                    instance=osp.join(rgb_dir, basename + ".jpg"),   # 图像文件的完整路径
                    is_metric=self.is_metric,                        # 是否是度量深度
                    is_video=ordered_video,                          # 是否是视频序列
                    quantile=np.array(1.0, dtype=np.float32),       # 分位数信息（可能用于深度缩放）
                    img_mask=img_mask,                               # 图像掩码
                    ray_mask=ray_mask,                               # 光线掩码
                    camera_only=False,                               # 是否只使用相机数据
                    depth_only=False,                                # 是否只使用深度数据
                    single_view=False,                               # 是否是单视图
                    reset=False,                                     # 是否重置（可能用于序列处理）
                )
            )
            # 这个字典包含了训练深度估计和多视图几何所需的所有信息
            # 特别适用于手术场景的3D重建和导航

        # ==================== 验证和返回 ====================
        assert len(views) == num_views
        # 确保返回的视图数量与请求的数量一致
        
        return views
        # 返回包含所有视图数据的列表


# ==================== 总结 ====================
"""
SCARED 数据集读取器的主要功能和特点：

1. **专业应用领域**：
   - 专门为内窥镜手术场景设计
   - 支持医疗机器人和手术导航应用
   - 适用于微创手术的3D重建

2. **数据集特点**：
   - 立体内窥镜相机数据
   - 真实的手术场景图像
   - 精确的深度信息（米为单位）
   - 关键帧采样策略

3. **深度处理优化**：
   - 专门针对手术场景的深度范围（<1米）
   - 处理反光表面和血液造成的深度噪声
   - 保持真实的物理尺度

4. **场景组织**：
   - 多个手术病例（dataset1-7）
   - 每个病例的关键帧序列
   - 支持时序分析和器械跟踪

5. **技术优势**：
   - 高精度的相机标定
   - 适合手术环境的图像预处理
   - 标准化的多视角接口
   - 支持实时手术导航

6. **应用场景**：
   - 手术机器人视觉系统
   - 内窥镜3D重建
   - 手术器械跟踪
   - 组织变形分析
   - 手术培训和模拟

这个数据集读取器为医疗机器人和计算机辅助手术提供了重要的数据支持，
展现了计算机视觉在医疗领域的专业化应用。
""" 