#!/usr/bin/env python3
"""
SCARED数据集类 - 无缩放版本
用于对比测试是否缩放影响训练效果
"""

import os.path as osp
import cv2
import numpy as np
import itertools
import os

from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class SCARED_NoScaling(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True  # SCARED是metric数据集
        self.max_interval = 4
        super().__init__(*args, **kwargs)
        assert self.split in ["train", "test", "val"]
        
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
            scene_dir = osp.join(root, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            
            if not os.path.exists(rgb_dir):
                continue
            
            # 获取所有图像文件的basename
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
            
            # 加载深度图（保持原始米单位，无缩放）
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            depthmap[~np.isfinite(depthmap)] = 0  # 处理无效值
            # 不进行20倍缩放！

            # 加载相机参数
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            pose_world2cam = cam["pose"].copy()

            # 转换为cam2world格式（不进行位移缩放）
            camera_pose = np.linalg.inv(pose_world2cam)
            # 不进行20倍缩放！

            intrinsics = cam["intrinsics"]
            
            # 图像裁剪和缩放
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # 使用标准掩码概率
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="SCARED",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".jpg"),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),  # 标准quantile
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


# 多视图版本
class SCARED_Multi_NoScaling(SCARED_NoScaling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, idx):
        views = super().__getitem__(idx)
        assert len(views) == self.num_views
        return views
