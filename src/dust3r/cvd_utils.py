import torch
import torch.nn.functional as F
from torch import Tensor
from .utils.camera import pose_encoding_to_camera

# This SSIM implementation is borrowed from DUST3R's original losses.py
# to keep cvd_utils.py self-contained with its dependencies.
class SSIM(torch.nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = torch.nn.AvgPool2d(3, 1)
        self.mu_y_pool   = torch.nn.AvgPool2d(3, 1)
        self.sig_x_pool  = torch.nn.AvgPool2d(3, 1)
        self.sig_y_pool  = torch.nn.AvgPool2d(3, 1)
        self.sig_xy_pool = torch.nn.AvgPool2d(3, 1)

        self.refl = torch.nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def get_pixel_grid(height, width, device):
    """Generates a pixel coordinate grid."""
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    grid = torch.stack([x_coords, y_coords], dim=0) # [2, H, W]
    return grid

def compute_rigid_flow(depth_source, pose_source_to_target, intrinsics, pixel_grid=None):
    """
    Computes rigid flow from depth, relative pose, and intrinsics.
    Args:
        depth_source: [B, H, W] source frame depth map.
        pose_source_to_target: [B, 4, 4] relative pose matrix from source to target.
        intrinsics: [B, 3, 3] camera intrinsics.
        pixel_grid: [2, H, W] optional pre-computed pixel grid.
    Returns:
        flow: [B, 2, H, W] rigid flow map.
        valid_mask: [B, 1, H, W] mask of valid projections.
    """
    B, H, W = depth_source.shape
    device = depth_source.device
    
    if pixel_grid is None:
        pixel_grid = get_pixel_grid(H, W, device)
    
    homo_coords = F.pad(pixel_grid, (0, 0, 0, 0, 0, 1), "constant", 1.0)
    homo_coords = homo_coords.view(3, -1).unsqueeze(0).repeat(B, 1, 1)
    
    inv_K = torch.inverse(intrinsics)
    cam_points = torch.bmm(inv_K, homo_coords)
    
    points_3d = cam_points * depth_source.view(B, 1, -1)
    
    points_3d_homo = F.pad(points_3d, (0, 0, 0, 1), "constant", 1.0)
    points_3d_target_homo = torch.bmm(pose_source_to_target, points_3d_homo)
    
    points_3d_target = points_3d_target_homo[:, :3, :]
    
    valid_depth_mask = points_3d_target[:, 2:3, :] > 1e-4
    
    pixel_coords_target = torch.bmm(intrinsics, points_3d_target)
    pixel_coords_target_uv = pixel_coords_target[:, :2, :] / (pixel_coords_target[:, 2:3, :] + 1e-7)
    
    source_coords_uv = pixel_grid.view(2, -1).unsqueeze(0).repeat(B, 1, 1)
    flow_vector = pixel_coords_target_uv - source_coords_uv
    
    flow_map = flow_vector.view(B, 2, H, W)
    valid_mask = valid_depth_mask.view(B, 1, H, W)
    
    return flow_map, valid_mask

def photometric_loss(img_source, img_target, depth_source, pose_source_to_target, intrinsics, pixel_grid=None):
    """
    Computes photometric consistency loss (L1 + SSIM).
    """
    B, _, H, W = img_source.shape
    device = img_source.device
    
    rigid_flow, valid_mask = compute_rigid_flow(depth_source, pose_source_to_target, intrinsics, pixel_grid)
    
    if pixel_grid is None:
        pixel_grid = get_pixel_grid(H, W, device)
        
    source_coords_uv = pixel_grid.unsqueeze(0).repeat(B, 1, 1, 1)
    
    sampling_grid = source_coords_uv + rigid_flow
    
    sampling_grid[:, 0] = (sampling_grid[:, 0] / (W - 1)) * 2.0 - 1.0
    sampling_grid[:, 1] = (sampling_grid[:, 1] / (H - 1)) * 2.0 - 1.0
    
    sampling_grid = sampling_grid.permute(0, 2, 3, 1)
    
    warped_source = F.grid_sample(
        img_source,
        sampling_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )
    
    out_of_bounds_mask = (sampling_grid[..., 0] < -1) | (sampling_grid[..., 0] > 1) | \
                         (sampling_grid[..., 1] < -1) | (sampling_grid[..., 1] > 1)
    
    mask = valid_mask & (~out_of_bounds_mask.unsqueeze(1))
    
    l1_loss = torch.abs(warped_source - img_target)
    l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-7)
    
    ssim = SSIM().to(device)
    ssim_loss_map = ssim(warped_source, img_target)
    ssim_loss = (ssim_loss_map * mask).sum() / (mask.sum() + 1e-7)
    
    loss = 0.85 * ssim_loss + 0.15 * l1_loss
    
    return loss, warped_source

def edge_aware_smoothness_loss(depth, image):
    """
    Computes an edge-aware depth smoothness loss.
    """
    def gradient(x):
        h_x, w_x = x.size()[-2:]
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy

    depth_dx, depth_dy = gradient(depth)
    image_dx, image_dy = gradient(image)

    weight_x = torch.exp(-torch.mean(torch.abs(image_dx), 1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(image_dy), 1, keepdim=True))
    
    smoothness_x = torch.abs(depth_dx) * weight_x
    smoothness_y = torch.abs(depth_dy) * weight_y
    
    return smoothness_x.mean() + smoothness_y.mean()
