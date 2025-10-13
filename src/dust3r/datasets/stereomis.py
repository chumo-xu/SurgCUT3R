import os
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class STEREOMIS_Multi(BaseMultiViewDataset):
    """
    Dataset loader for preprocessed StereoMIS sequences.

    Expected directory layout:
        ROOT/
          <sequence_name>/
            rgb/   *.jpg
            depth/ *.npy  (metric depth preferred)
            cam/   *.npz  (keys: 'pose' (4x4 cam2world), 'intrinsics' (3x3))

    Notes:
    - This dataset does not use train/val/test subfolders by default. The `split`
      argument is accepted for API compatibility but ignored (all sequences under ROOT
      are used).
    - By default, depth and pose translation are scaled by 20.0x to match
      SCARED_New15 behavior (set scale_depth_pose to override if needed).
    """

    def __init__(self, *args, ROOT, scale_depth_pose: float = 20.0, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 12
        self.scale_depth_pose = float(scale_depth_pose)
        super().__init__(*args, **kwargs)
        # Load once
        self.loaded_data = self._load_data()

    def _load_data(self):
        root = self.ROOT  # split ignored; use all sequences under ROOT

        self.scenes = []
        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(sorted(os.listdir(root))):
            scene_dir = osp.join(root, scene)
            if not osp.isdir(scene_dir):
                continue
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")
            if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) and osp.isdir(cam_dir)):
                # skip directories that do not look like a sequence
                continue

            # basenames without extension (assumed 6-digit filenames)
            basenames = sorted([f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
            num_imgs = len(basenames)
            if num_imgs == 0:
                continue

            # construct global indexing
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            if num_imgs < cut_off:
                print(f"Skipping {scene} (too few images: {num_imgs})")
                continue

            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            offset += num_imgs
            j += 1

        # store
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
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # RGB
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))

            # Depth
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            depthmap[~np.isfinite(depthmap)] = 0
            if self.scale_depth_pose != 1.0:
                depthmap = depthmap * self.scale_depth_pose

            # Camera
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"].copy()  # expected cam2world
            intrinsics = cam["intrinsics"].copy()
            if self.scale_depth_pose != 1.0:
                camera_pose[:3, 3] = camera_pose[:3, 3] * self.scale_depth_pose

            # Debug print only once
            if not hasattr(self, "_pose_debug_printed"):
                print("[StereoMIS] 加载与尺度设置:")
                print(f"  位姿(cam2world)位移(已缩放): {camera_pose[:3, 3]}")
                print(f"  深度/位移缩放因子: {self.scale_depth_pose} ✅ (在读取器中固定应用)")
                self._pose_debug_printed = True

            # Crop+resize
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # Masks
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[1.0, 0.0, 0.0])

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="StereoMIS",
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

