import os.path as osp
import os
import sys
import numpy as np

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2

class Scared_Multi(BaseMultiViewDataset):
    """
    Dataset loader for the preprocessed Scared1-1 dataset.
    The dataset is expected to be in a directory with the following structure:
    - root/
      - camera/ (contains .npz files with intrinsics and extrinsics)
      - color/  (contains .png color images)
      - depth/  (contains .npy depth maps in meters)
    """
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.is_metric = True  # Depth is in meters
        super().__init__(*args, **kwargs)
        self.frames = self._load_data()

    def _load_data(self):
        """Scans the color directory to find all available frames."""
        color_dir = osp.join(self.ROOT, "color")
        frames = sorted([osp.splitext(f)[0] for f in os.listdir(color_dir) if f.endswith('.png')])
        print(f"Found {len(frames)} frames in {self.ROOT}")
        return frames

    def __len__(self):
        """Returns the total number of frames in the dataset."""
        return len(self.frames)

    def get_image_num(self):
        return len(self)

    def _get_views(self, idx, resolution, rng, num_views):
        """
        Get a sequence of views. For simplicity, we'll start with a simple sequential sampling.
        This can be evolved to more complex sampling strategies later.
        """
        # For now, we sample `num_views` consecutive frames starting from idx
        # Making sure we don't go out of bounds
        start_idx = idx
        end_idx = min(start_idx + num_views, len(self.frames))
        if end_idx - start_idx < num_views:
            start_idx = end_idx - num_views

        image_idxs = list(range(start_idx, end_idx))
        
        views = []
        for view_offset, view_idx in enumerate(image_idxs):
            base_name = self.frames[view_idx]

            # Paths to data files
            color_path = osp.join(self.ROOT, "color", f"{base_name}.png")
            depth_path = osp.join(self.ROOT, "depth", f"{base_name}.npy")
            camera_path = osp.join(self.ROOT, "camera", f"{base_name}.npz")

            # Load data
            rgb_image = imread_cv2(color_path)
            depthmap = np.load(depth_path)
            camera_data = np.load(camera_path)
            camera_intrinsics = camera_data['intrinsics']
            camera_pose = camera_data['extrinsics']

            # Set invalid depth to 0
            depthmap[~np.isfinite(depthmap)] = 0

            # This part can be adapted from ARKitScenes if needed
            rgb_image, depthmap, camera_intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, view_offset, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=camera_intrinsics.astype(np.float32),
                    dataset="scared",
                    label=f"scared_{base_name}",
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=True, # It's a sequence
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        
        # If not enough views were gathered, pad by repeating the last one
        while len(views) < num_views:
            views.append(views[-1].copy())

        assert len(views) == num_views
        return views 