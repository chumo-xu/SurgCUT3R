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


class SCARED_Multinew10(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
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
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            if num_imgs < cut_off:
                print(f"Skipping {scene} (too few images: {num_imgs})")
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

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
        )
        image_idxs = np.array(all_image_ids)[pos]

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
            pose_world2cam = cam["pose"].copy()  # 原始位姿是world2cam格式

            # 转换为CUT3R期望的cam2world格式
            camera_pose = np.linalg.inv(pose_world2cam)  # world2cam -> cam2world

            # 调试信息：验证位姿转换（仅在第一次加载时打印）
            if hasattr(self, '_pose_debug_printed') is False:
                print(f"[SCARED数据集] 位姿转换验证:")
                print(f"  原始位姿(world2cam)位移: {pose_world2cam[:3, 3]}")
                print(f"  转换后(cam2world)位移: {camera_pose[:3, 3]}")
                print(f"  位姿格式已从world2cam转换为cam2world ✅")
                self._pose_debug_printed = True

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
                    dataset="SCARED",  # 数据集标识
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