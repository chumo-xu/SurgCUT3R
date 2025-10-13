import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class SCARED_Shuffled(BaseMultiViewDataset):
    """
    SCARED数据集的Shuffled版本
    
    主要改进：
    1. 在每个keyframe内进行shuffle，避免连续帧导致的位姿变化过小
    2. 只保留4的倍数帧，确保批次完整性
    3. 增加位姿变化，防止模型陷入位姿不变的误区
    """
    
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 50  # 🔥 大幅增加最大间隔，允许更大的位姿变化
        super().__init__(*args, **kwargs)
        assert self.split in ["train", "test", "val"]
        
        # SCARED数据集的所有序列名称
        self.scenes_to_use = [
            # Dataset 1
            "dataset1_keyframe1",
            "dataset1_keyframe2", 
            "dataset1_keyframe3",
            # Dataset 2
            "dataset2_keyframe1",
            "dataset2_keyframe2",
            "dataset2_keyframe3",
            "dataset2_keyframe4",
            # Dataset 3
            "dataset3_keyframe1",
            "dataset3_keyframe2", 
            "dataset3_keyframe3",
            "dataset3_keyframe4",
            # Dataset 6
            "dataset6_keyframe1",
            "dataset6_keyframe2",
            "dataset6_keyframe3", 
            "dataset6_keyframe4",
            # Dataset 7
            "dataset7_keyframe1",
            "dataset7_keyframe2",
            "dataset7_keyframe3",
            "dataset7_keyframe4",
        ]
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        """
        加载数据，确保每个keyframe只保留4的倍数帧
        """
        root = os.path.join(self.ROOT, split)
        self.scenes = []

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(os.listdir(root)):
            if scene not in self.scenes_to_use:
                continue
            scene_dir = osp.join(root, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            
            # 获取所有图像文件的basename（去掉.jpg后缀）
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
            )
            
            # 🔥 关键修改：只保留4的倍数帧
            total_imgs = len(basenames)
            # 计算最大的4的倍数
            max_multiple_of_4 = (total_imgs // 4) * 4
            
            if max_multiple_of_4 < 4:
                print(f"Skipping {scene} (insufficient frames for 4-multiple: {total_imgs})")
                continue
            
            # 只保留前N*4帧
            basenames = basenames[:max_multiple_of_4]
            num_imgs = len(basenames)
            
            print(f"Scene {scene}: {total_imgs} -> {num_imgs} frames (4-multiple)")
            
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            if num_imgs < cut_off:
                print(f"Skipping {scene} (too few images after filtering: {num_imgs})")
                continue

            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            # offset groups
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

        print(f"📊 数据加载统计:")
        print(f"   总场景数: {len(self.scenes)}")
        print(f"   总图像数: {len(self.images)}")
        print(f"   起始点数: {len(self.start_img_ids)}")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        """
        获取视图，使用改进的shuffle策略增加位姿变化
        """
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]

        # 🔥 关键修改：使用极激进的shuffle参数，最大化位姿变化
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=0.0,           # 完全禁用连续序列，100%随机选择
            fix_interval_prob=0.0,    # 完全禁用固定间隔，100%随机间隔
            block_shuffle=None,       # 完全随机打乱，不使用块结构
        )
        image_idxs = np.array(all_image_ids)[pos]

        # 🔍 调试信息：打印选择的帧索引，验证shuffle是否生效
        if idx % 100 == 0:  # 每100个样本打印一次
            scene_id = self.sceneids[start_id]
            scene_name = self.scenes[scene_id]
            print(f"DEBUG [idx={idx}] Scene: {scene_name}")
            print(f"  start_id: {start_id}, all_image_ids range: {min(all_image_ids)}-{max(all_image_ids)}")
            print(f"  selected positions: {pos}")
            print(f"  selected image_idxs: {image_idxs}")
            print(f"  ordered_video: {ordered_video}")
            print(f"  frame intervals: {[image_idxs[i+1] - image_idxs[i] for i in range(len(image_idxs)-1)]}")
            print("---")

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # 加载RGB图像
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            
            # 加载深度图（单位：米）
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            depthmap[~np.isfinite(depthmap)] = 0  # 处理无效值
            
            # SCARED深度单位是米，缩放20倍以匹配预训练权重的尺度
            depthmap = depthmap * 20.0  # 深度缩放20倍
            #depthmap[depthmap > 50.0] = 0.0  # 过滤超过50米的深度值（原1米*50）

            # 加载相机参数
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"].copy()  # 复制以避免修改原始数据
            # 缩放位移向量T（4x4矩阵的最后一列前3个元素）
            camera_pose[:3, 3] = camera_pose[:3, 3] * 20.0  # T向量缩放20倍
            intrinsics = cam["intrinsics"]
            
            # 图像裁剪和缩放
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # 生成图像掩码和射线掩码
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.9, 0.05, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="SCARED_Shuffled",  # 🔥 修改数据集标识
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".jpg"),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        assert len(views) == num_views
        return views
