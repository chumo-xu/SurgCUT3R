from copy import copy, deepcopy
import torch
import torch.nn as nn

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import (
    inv,
    geotrf,
    normalize_pointcloud,
    normalize_pointcloud_group,
)
from dust3r.utils.geometry import (
    get_group_pointcloud_depth,
    get_group_pointcloud_center_scale,
    weighted_procrustes,
)
from gsplat import rasterization
import numpy as np
import lpips
from dust3r.utils.camera import (
    pose_encoding_to_camera,
    camera_to_pose_encoding,
    relative_pose_absT_quatR,
)
import torch.nn.functional as F


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    """L-norm loss"""

    def forward(self, a, b):
        assert (
            a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3
        ), f"Bad shape = {a.shape}"
        dist = self.distance(a, b)
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss(LLoss):
    """Euclidean distance between 3d points"""

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class MSELoss(LLoss):
    def distance(self, a, b):
        return (a - b) ** 2


MSE = MSELoss()


class Criterion(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(
            criterion, BaseCriterion
        ), f"{criterion} is not a proper criterion!"
        self.criterion = copy(criterion)

    def get_name(self):
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class RGBLoss(Criterion, MultiLoss):
    def __init__(self, criterion):
        super().__init__(criterion)
        self.ssim = SSIM()

    def img_loss(self, a, b):
        return self.criterion(a, b)

    def compute_loss(self, gts, preds, **kw):
        gt_rgbs = [gt["img"].permute(0, 2, 3, 1) for gt in gts]
        pred_rgbs = [pred["rgb"] for pred in preds]
        ls = [
            self.img_loss(pred_rgb, gt_rgb)
            for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs)
        ]
        details = {}
        self_name = type(self).__name__
        for i, l in enumerate(ls):
            details[self_name + f"_rgb/{i+1}"] = float(l)
            details[f"pred_rgb_{i+1}"] = pred_rgbs[i]
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details


class DepthScaleShiftInvLoss(BaseCriterion):
    """scale and shift invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x, mask):
        x_valid = x[mask]
        splits = mask.sum(dim=(1, 2)).tolist()
        x_valid_list = torch.split(x_valid, splits)
        shift = [x.mean() for x in x_valid_list]
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]
        scale = [x.abs().mean() for x in x_valid_centered]
        scale = torch.stack(scale)
        shift = torch.stack(shift)
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(self, pred, gt, mask):
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)
        return torch.abs((pred - gt)[mask])


class ScaleInvLoss(BaseCriterion):
    """scale invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 4, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, pred, gt, mask):
        pred_norm_factor = (torch.norm(pred, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        gt_norm_factor = (torch.norm(gt, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        pred = pred / pred_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        gt = gt / gt_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        return torch.norm(pred - gt, dim=-1)[mask]


class Regr3DPose(Criterion, MultiLoss):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
    ):
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_norm_factor_point_cloud(
        self, pts_self, pts_cross, valids, conf_self, conf_cross, norm_self_only=False
    ):
        if norm_self_only:
            norm_factor = normalize_pointcloud_group(
                pts_self, self.norm_mode, valids, conf_self, ret_factor_only=True
            )
        else:
            pts = [torch.cat([x, y], dim=2) for x, y in zip(pts_self, pts_cross)]
            valids = [torch.cat([x, x], dim=2) for x in valids]
            confs = [torch.cat([x, y], dim=2) for x, y in zip(conf_self, conf_cross)]
            norm_factor = normalize_pointcloud_group(
                pts, self.norm_mode, valids, confs, ret_factor_only=True
            )
        return norm_factor

    def get_norm_factor_poses(self, gt_trans, pr_trans, not_metric_mask):

        if self.norm_mode and not self.gt_scale:
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
            norm_factor_gt = torch.ones(
                len(gt_trans), dtype=gt_trans[0].dtype, device=gt_trans[0].device
            )

        norm_factor_pr = norm_factor_gt.clone()
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

    def get_all_pts3d(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
        camera1=None,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"]) if camera1 is None else inv(camera1)
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self,
                gt_pts_cross,
                valids,
                conf_self,
                conf_cross,
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            norm_factor_pr_not_metric = self.get_norm_factor_point_cloud(
                [pr_pt_self[not_metric_mask] for pr_pt_self in pr_pts_self],
                [pr_pt_cross[not_metric_mask] for pr_pt_cross in pr_pts_cross],
                [valid[not_metric_mask] for valid in valids],
                [conf[not_metric_mask] for conf in conf_self],
                [conf[not_metric_mask] for conf in conf_cross],
                norm_self_only=norm_self_only,
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, not_metric_mask
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(gt_trans, pr_trans, not_metric_mask)
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )

        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]
        pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
            pose_norm_factor_pr.squeeze() > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {},
        )

    def get_all_pts3d_with_scale_loss(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"])
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self[:1],
                gt_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        if self.norm_mode:
            norm_factor_pr = self.get_norm_factor_point_cloud(
                pr_pts_self[:1],
                pr_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            raise NotImplementedError
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pts_scale_loss = torch.abs(
                norm_factor_pr[~not_metric_mask] - norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pts_scale_loss = 0.0

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, torch.ones_like(not_metric_mask)
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(
                    gt_trans, pr_trans, torch.ones_like(not_metric_mask)
                )
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pose_scale_loss = torch.abs(
                pose_norm_factor_pr[~not_metric_mask]
                - pose_norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pose_scale_loss = 0.0
        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]

        pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
            pose_norm_factor_pr.squeeze() > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {"scale_loss": pose_scale_loss + pts_scale_loss},
        )

    def compute_relative_pose_loss(
        self, gt_trans, gt_quats, pr_trans, pr_quats, masks=None
    ):
        if masks is None:
            masks = torch.ones(len(gt_trans), dtype=torch.bool, device=gt_trans.device)
        gt_trans_matrix1 = gt_trans[:, :, None, :].repeat(1, 1, gt_trans.shape[1], 1)[
            masks
        ]
        gt_trans_matrix2 = gt_trans[:, None, :, :].repeat(1, gt_trans.shape[1], 1, 1)[
            masks
        ]
        gt_quats_matrix1 = gt_quats[:, :, None, :].repeat(1, 1, gt_quats.shape[1], 1)[
            masks
        ]
        gt_quats_matrix2 = gt_quats[:, None, :, :].repeat(1, gt_quats.shape[1], 1, 1)[
            masks
        ]
        pr_trans_matrix1 = pr_trans[:, :, None, :].repeat(1, 1, pr_trans.shape[1], 1)[
            masks
        ]
        pr_trans_matrix2 = pr_trans[:, None, :, :].repeat(1, pr_trans.shape[1], 1, 1)[
            masks
        ]
        pr_quats_matrix1 = pr_quats[:, :, None, :].repeat(1, 1, pr_quats.shape[1], 1)[
            masks
        ]
        pr_quats_matrix2 = pr_quats[:, None, :, :].repeat(1, pr_quats.shape[1], 1, 1)[
            masks
        ]

        gt_rel_trans, gt_rel_quats = relative_pose_absT_quatR(
            gt_trans_matrix1, gt_quats_matrix1, gt_trans_matrix2, gt_quats_matrix2
        )
        pr_rel_trans, pr_rel_quats = relative_pose_absT_quatR(
            pr_trans_matrix1, pr_quats_matrix1, pr_trans_matrix2, pr_quats_matrix2
        )
        rel_trans_err = torch.norm(gt_rel_trans - pr_rel_trans, dim=-1)
        rel_quats_err = torch.norm(gt_rel_quats - pr_rel_quats, dim=-1)
        return rel_trans_err.mean() + rel_quats_err.mean()

    def compute_pose_loss(self, gt_poses, pred_poses, masks=None):
        """
        gt_pose: list of (Bx3, Bx4)
        pred_pose: list of (Bx3, Bx4)
        masks: None, or B
        """
        gt_trans = torch.stack([gt[0] for gt in gt_poses], dim=1)  # BxNx3
        gt_quats = torch.stack([gt[1] for gt in gt_poses], dim=1)  # BXNX3
        pred_trans = torch.stack([pr[0] for pr in pred_poses], dim=1)  # BxNx4
        pred_quats = torch.stack([pr[1] for pr in pred_poses], dim=1)  # BxNx4
        if masks == None:
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1).mean()
                + torch.norm(pred_quats - gt_quats, dim=-1).mean()
            )
        else:
            # 增加一个健壮的检查，同时处理列表和张量类型的 masks
            is_empty = False
            if isinstance(masks, torch.Tensor):
                if not torch.any(masks):
                    is_empty = True
            elif not any(masks):
                is_empty = True
        
            if is_empty:
                return torch.tensor(0.0, device=pred_trans.device)

            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1)[masks].mean()
                + torch.norm(pred_quats - gt_quats, dim=-1)[masks].mean()
            )

        return pose_loss

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_self,
            gt_pts_cross,
            pred_pts_self,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            # masks are overwritten taking into account self view losses
            ls_self, masks = self.criterion(
                pred_pts_self, gt_pts_self, masks, gts[0]["quantile"]
            )
        else:
            ls_self = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(pred_pts_self, gt_pts_self, masks)
            ]

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_self):
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}
        for i in range(len(ls_self)):
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        # cross view loss and details
        camera_only = gts[0]["camera_only"]
        pred_pts_cross = [pred_pts[~camera_only] for pred_pts in pred_pts_cross]
        gt_pts_cross = [gt_pts[~camera_only] for gt_pts in gt_pts_cross]
        masks_cross = [mask[~camera_only] for mask in masks]
        skys_cross = [sky[~camera_only] for sky in skys]

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            ls_cross, _ = self.criterion(
                pred_pts_cross, gt_pts_cross, masks_cross, None
            )
        else:
            ls_cross = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(
                    pred_pts_cross, gt_pts_cross, masks_cross
                )
            ]

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_self + ls_cross
        masks = masks + masks_cross
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        # ---- Add smoothness loss here ----
        # 1. Get predicted depth and image from the first view (anchor view)
        # Using .detach() on depth because smoothness is a regularization term,
        # it should not affect the 3D point regression task directly.
        depth = preds[0]["pts3d_in_self_view"][..., 2].detach() # Shape (B, H, W)
        image = gts[0]["img"] # Shape (B, C, H, W)

        # 2. Convert depth to disparity and normalize
        disp = 1.0 / (depth + 1e-7)
        mean_disp = disp.mean(1, True).mean(2, True)
        norm_disp = disp / (mean_disp + 1e-7)
        
        # Add channel dimension to disparity map for get_smooth_loss
        norm_disp = norm_disp.unsqueeze(1) # Shape (B, 1, H, W)

        # 3. Compute smoothness loss
        # The weight 0.1 is a starting point, you should add it to your config file
        smoothness_loss = get_smooth_loss(norm_disp, image) * 0.1
        details["smoothness_loss"] = float(smoothness_loss)
        # ----------------------------------
        
        return Sum(*list(zip(ls, masks))), (details | monitoring)


class Regr3DPoseBatchList(Regr3DPose):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
    ):
        super().__init__(
            criterion, norm_mode, gt_scale, sky_loss_value, max_metric_scale
        )
        self.depth_only_criterion = DepthScaleShiftInvLoss()
        self.single_view_criterion = ScaleInvLoss()

    def reorg(self, ls_b, masks_b):
        ids_split = [mask.sum(dim=(1, 2)) for mask in masks_b]
        ls = [[] for _ in range(len(masks_b[0]))]
        for i in range(len(ls_b)):
            ls_splitted_i = torch.split(ls_b[i], ids_split[i].tolist())
            for j in range(len(masks_b[0])):
                ls[j].append(ls_splitted_i[j])
        ls = [torch.cat(l) for l in ls]
        return ls

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_self,
            gt_pts_cross,
            pred_pts_self,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        camera_only = gts[0]["camera_only"]
        depth_only = gts[0]["depth_only"]
        single_view = gts[0]["single_view"]
        is_metric = gts[0]["is_metric"]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            raise NotImplementedError
        else:
            # list [(B, h, w, 3)] x num_views -> list [num_views, h, w, 3] x B
            gt_pts_self_b = torch.unbind(torch.stack(gt_pts_self, dim=1), dim=0)
            pred_pts_self_b = torch.unbind(torch.stack(pred_pts_self, dim=1), dim=0)
            masks_b = torch.unbind(torch.stack(masks, dim=1), dim=0)
            ls_self_b = []
            for i in range(len(gt_pts_self_b)):
                if depth_only[
                    i
                ]:  # if only have relative depth, no intrinsics or anything
                    ls_self_b.append(
                        self.depth_only_criterion(
                            pred_pts_self_b[i][..., -1],
                            gt_pts_self_b[i][..., -1],
                            masks_b[i],
                        )
                    )
                elif (
                    single_view[i] and not is_metric[i]
                ):  # if single view, with intrinsics and not metric
                    ls_self_b.append(
                        self.single_view_criterion(
                            pred_pts_self_b[i], gt_pts_self_b[i], masks_b[i]
                        )
                    )
                else:  # if multiple view, or metric single view
                    ls_self_b.append(
                        self.criterion(
                            pred_pts_self_b[i][masks_b[i]], gt_pts_self_b[i][masks_b[i]]
                        )
                    )
            ls_self = self.reorg(ls_self_b, masks_b)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_self):
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}
        for i in range(len(ls_self)):
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            raise NotImplementedError
        else:
            gt_pts_cross_b = torch.unbind(
                torch.stack(gt_pts_cross, dim=1)[~camera_only], dim=0
            )
            pred_pts_cross_b = torch.unbind(
                torch.stack(pred_pts_cross, dim=1)[~camera_only], dim=0
            )
            masks_cross_b = torch.unbind(torch.stack(masks, dim=1)[~camera_only], dim=0)
            ls_cross_b = []
            for i in range(len(gt_pts_cross_b)):
                if depth_only[~camera_only][i]:
                    ls_cross_b.append(
                        self.depth_only_criterion(
                            pred_pts_cross_b[i][..., -1],
                            gt_pts_cross_b[i][..., -1],
                            masks_cross_b[i],
                        )
                    )
                elif single_view[~camera_only][i] and not is_metric[~camera_only][i]:
                    ls_cross_b.append(
                        self.single_view_criterion(
                            pred_pts_cross_b[i], gt_pts_cross_b[i], masks_cross_b[i]
                        )
                    )
                else:
                    ls_cross_b.append(
                        self.criterion(
                            pred_pts_cross_b[i][masks_cross_b[i]],
                            gt_pts_cross_b[i][masks_cross_b[i]],
                        )
                    )
            ls_cross = self.reorg(ls_cross_b, masks_cross_b)

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

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_self + ls_cross
        masks = masks + masks_cross
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )
        pose_masks = pose_masks * gts[i]["img_mask"]
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        # ---- Add smoothness loss here ----
        # 1. Get predicted depth and image from the first view (anchor view)
        # Using .detach() on depth because smoothness is a regularization term,
        # it should not affect the 3D point regression task directly.
        depth = preds[0]["pts3d_in_self_view"][..., 2].detach() # Shape (B, H, W)
        image = gts[0]["img"] # Shape (B, C, H, W)

        # 2. Convert depth to disparity and normalize
        disp = 1.0 / (depth + 1e-7)
        mean_disp = disp.mean(1, True).mean(2, True)
        norm_disp = disp / (mean_disp + 1e-7)
        
        # Add channel dimension to disparity map for get_smooth_loss
        norm_disp = norm_disp.unsqueeze(1) # Shape (B, 1, H, W)

        # 3. Compute smoothness loss
        # The weight 0.1 is a starting point, you should add it to your config file
        smoothness_loss = get_smooth_loss(norm_disp, image) * 0.1
        details["smoothness_loss"] = float(smoothness_loss)
        # ----------------------------------
        
        return Sum(*list(zip(ls, masks))), (details | monitoring)


class ConfLoss(MultiLoss):
    """Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self):
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gts, preds, **kw):
        # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds, **kw)
        if "is_self" in details and "img_ids" in details:
            is_self = details["is_self"]
            img_ids = details["img_ids"]
        else:
            is_self = [False] * len(losses_and_masks)
            img_ids = list(range(len(losses_and_masks)))

        # weight by confidence
        conf_losses = []

        for i in range(len(losses_and_masks)):
            pred = preds[img_ids[i]]
            conf_key = "conf_self" if is_self[i] else "conf"
            if not is_self[i]:
                camera_only = gts[0]["camera_only"]
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][~camera_only][losses_and_masks[i][1]]
                )
            else:
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][losses_and_masks[i][1]]
                )

            conf_loss = losses_and_masks[i][0] * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)

            if is_self[i]:
                details[self.get_name() + f"_conf_loss_self/{img_ids[i]+1}"] = float(
                    conf_loss
                )
            else:
                details[self.get_name() + f"_conf_loss/{img_ids[i]+1}"] = float(
                    conf_loss
                )

        details.pop("is_self", None)
        details.pop("img_ids", None)

        final_loss = sum(conf_losses) / len(conf_losses) * 2.0
        if "pose_loss" in details:
            #final_loss = (final_loss + details["pose_loss"].clip(max=0.3) * 5.0)  # , details
            final_loss = final_loss + details["pose_loss"] * 2.0
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]

        # ---- Add smoothness loss here ----
        # 1. Get predicted depth and image from the first view (anchor view)
        # Using .detach() on depth because smoothness is a regularization term,
        # it should not affect the 3D point regression task directly.
        depth = preds[0]["pts3d_in_self_view"][..., 2].detach() # Shape (B, H, W)
        image = gts[0]["img"] # Shape (B, C, H, W)

        # 2. Convert depth to disparity and normalize
        disp = 1.0 / (depth + 1e-7)
        mean_disp = disp.mean(1, True).mean(2, True)
        norm_disp = disp / (mean_disp + 1e-7)
        
        # Add channel dimension to disparity map for get_smooth_loss
        norm_disp = norm_disp.unsqueeze(1) # Shape (B, 1, H, W)

        # 3. Compute smoothness loss
        # The weight 0.1 is a starting point, you should add it to your config file
        smoothness_loss = get_smooth_loss(norm_disp, image) * 0.1
        final_loss = final_loss + smoothness_loss
        details["smoothness_loss"] = float(smoothness_loss)
        # ----------------------------------
        
        return final_loss, details


class Regr3DPose_ScaleInv(Regr3DPose):
    """Same than Regr3D but invariant to depth shift.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gts, preds):
        # compute depth-normalized points
        (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = super().get_all_pts3d(gts, preds)

        # measure scene scale
        _, gt_scale_self = get_group_pointcloud_center_scale(gt_pts_self, masks)
        _, pred_scale_self = get_group_pointcloud_center_scale(pr_pts_self, masks)

        _, gt_scale_cross = get_group_pointcloud_center_scale(gt_pts_cross, masks)
        _, pred_scale_cross = get_group_pointcloud_center_scale(pr_pts_cross, masks)

        # prevent predictions to be in a ridiculous range
        pred_scale_self = pred_scale_self.clip(min=1e-3, max=1e3)
        pred_scale_cross = pred_scale_cross.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pr_pts_self = [
                pr_pt_self * gt_scale_self / pred_scale_self
                for pr_pt_self in pr_pts_self
            ]
            pr_pts_cross = [
                pr_pt_cross * gt_scale_cross / pred_scale_cross
                for pr_pt_cross in pr_pts_cross
            ]
        else:
            gt_pts_self = [gt_pt_self / gt_scale_self for gt_pt_self in gt_pts_self]
            gt_pts_cross = [
                gt_pt_cross / gt_scale_cross for gt_pt_cross in gt_pts_cross
            ]
            pr_pts_self = [pr_pt_self / pred_scale_self for pr_pt_self in pr_pts_self]
            pr_pts_cross = [
                pr_pt_cross / pred_scale_cross for pr_pt_cross in pr_pts_cross
            ]

        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        )


class ConfLoss_Decoupled(MultiLoss):
    """
    解耦的置信度损失 - 将pose_loss从ConfLoss中分离出来

    这个类是ConfLoss的修改版本，主要变化：
    1. 不再自动添加pose_loss到最终损失中
    2. 将pose_loss作为独立项返回到details中
    3. 允许外部损失函数独立控制pose_loss的权重

    使用场景：
    - 需要独立控制位姿损失权重时
    - 实现复杂的损失组合策略时
    - 避免位姿损失的"偷懒"现象时
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0, "alpha必须为正数"
        self.alpha = alpha
        # 将像素级损失设置为"none"模式，获取每个像素的损失值
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self):
        """获取损失函数名称"""
        return f"ConfLoss_Decoupled({self.pixel_loss})"

    def get_conf_log(self, x):
        """
        获取置信度和其对数

        参数:
            x: 置信度张量

        返回:
            (conf, log_conf): 置信度和其对数
        """
        return x, torch.log(x)

    def compute_loss(self, gts, preds, **kw):
        """
        计算解耦的置信度损失

        与原始ConfLoss的主要区别：
        1. 不自动添加pose_loss到最终损失
        2. 将pose_loss保留在details中供外部使用

        参数:
            gts: 真实数据列表
            preds: 预测数据列表
            **kw: 其他关键字参数

        返回:
            (conf_loss_only, details): 仅置信度损失和详细信息
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

        # 5. 计算最终的置信度损失（不包含pose_loss）
        final_conf_loss = sum(conf_losses) / len(conf_losses) * 2.0

        # 6. 保留pose_loss和scale_loss在details中，但不添加到最终损失
        # 这些损失将由外部的DecoupledTotalLoss来处理

        return final_conf_loss, details




class DINOv2FeatureExtractor(nn.Module):
    """
    轻量封装的 DINOv2 稠密特征提取器（冻结）。

    优先顺序：
    1) torchvision.models.dinov2（如果可用）
    2) timm 的 vit_base_patch14_dinov2（如果可用）
    3) 安全降级：返回简单的L2归一化RGB特征（不会报错，但效果较弱）
    """

    def __init__(self, out_stride=14, device=None):
        super().__init__()
        self.out_stride = out_stride
        self.device = device
        self.model = None
        self.backend = None

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # 尝试 torchvision 的 dinov2
        try:
            from torchvision.models import dinov2 as tv_dinov2  # type: ignore

            # 使用 ViT-B/14
            self.model = tv_dinov2.dinov2_vitb14(pretrained=True)
            self.backend = "torchvision"
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        except Exception:
            # 尝试 timm
            try:
                import timm  # type: ignore

                self.model = timm.create_model(
                    "vit_base_patch14_dinov2", pretrained=True
                )
                self.backend = "timm"
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()
            except Exception:
                self.model = None
                self.backend = "fallback"

        if device is not None:
            self.to(device)

    @torch.no_grad()
    def forward(self, x):
        """
        输入: x in [0,1], shape (B,3,H,W)
        输出: 稠密描述子 (B,C,Hf,Wf)，已L2归一化
        """
        if self.backend == "torchvision":
            # torchvision 的 dinov2 接口：返回 patch tokens 需使用 get_intermediate_layers
            x_norm = (x - self.mean) / self.std
            # 获取最后一层的所有tokens（包含cls）
            feats = self.model.get_intermediate_layers(x_norm, n=1)[0]  # (B, 1+HW/14^2, C)
            B, N, C = feats.shape
            # 去掉 cls token
            feats = feats[:, 1:, :]
            # 估计网格分辨率
            Hf = x.shape[-2] // 14
            Wf = x.shape[-1] // 14
            feats = feats.transpose(1, 2).reshape(B, C, Hf, Wf)
            feats = torch.nn.functional.normalize(feats, dim=1)
            return feats

        if self.backend == "timm":
            # timm ViT: 使用 forward_features 拿到 patch tokens
            x_norm = (x - self.mean) / self.std
            out = self.model.forward_features(x_norm)
            # 兼容不同返回类型
            if isinstance(out, dict) and "x_norm_patchtokens" in out:
                tokens = out["x_norm_patchtokens"]  # (B, HW, C)
            elif isinstance(out, (list, tuple)):
                tokens = out[-1]
            else:
                # 有些版本直接返回 (B, HW+1, C)
                tokens = out
            if tokens.dim() == 3:
                B, N, C = tokens.shape
                # 去掉 cls（如果存在）
                if N > (x.shape[-2] // 14) * (x.shape[-1] // 14):
                    tokens = tokens[:, 1:, :]
                Hf = x.shape[-2] // 14
                Wf = x.shape[-1] // 14
                feats = tokens.transpose(1, 2).reshape(B, C, Hf, Wf)
            else:
                # 兜底：若已是特征图
                feats = tokens
            feats = torch.nn.functional.normalize(feats, dim=1)
            return feats

        # 安全降级：返回归一化后的RGB作为弱特征
        feats = torch.nn.functional.interpolate(x, scale_factor=1 / 14, mode="bilinear", align_corners=False)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats


class DINOFlowLoss(MultiLoss):
    """
    基于 DINOv2 特征的伪光流一致性，用于强化位姿估计。

    核心：
    1) 用冻结的 DINOv2 提取两帧稠密特征 → 互为最近邻（局部窗口）建立伪光流 O~
    2) 用预测位姿+（深度 stop-grad）诱导的像素位移 f_pose 与 O~ 对齐
    3) 仅在可靠掩码内计算（互为最近邻、一致性与可见性）
    """

    def __init__(
        self,
        window_size: int = 9,
        sim_threshold: float = 0.6,
        fb_thresh: float = 1.5,
        stop_grad_depth: bool = True,
        feature_extractor: nn.Module | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.sim_threshold = sim_threshold
        self.fb_thresh = fb_thresh
        self.stop_grad_depth = stop_grad_depth
        self.extractor = feature_extractor if feature_extractor is not None else DINOv2FeatureExtractor()

    def get_name(self):
        return "DINOFlowLoss"

    @staticmethod
    def _mutual_nn_local_flow(Fi, Fj, window_size: int):
        """
        在局部 k×k 窗内做互为最近邻，返回：
        - 伪光流 i->j 与 j->i（在特征分辨率上，单位：像素）
        - 相似度最大值与互为最近邻掩码
        """
        B, C, Hf, Wf = Fi.shape
        r = window_size // 2

        # 归一化
        Fi = torch.nn.functional.normalize(Fi, dim=1)
        Fj = torch.nn.functional.normalize(Fj, dim=1)

        # 使用 unfold 构造 j 的局部窗口特征: [B, C*k*k, Hf*Wf]
        Fj_unf = F.unfold(Fj, kernel_size=window_size, padding=r)  # [B, C*k*k, Hf*Wf]
        Fj_unf = Fj_unf.view(B, C, window_size * window_size, Hf, Wf)

        # i->j: 与局部窗口计算相似度
        Fi_e = Fi.unsqueeze(2)  # [B, C, 1, Hf, Wf]
        sim_ij = (Fi_e * Fj_unf).sum(dim=1)  # [B, k*k, Hf, Wf]
        maxv_ij, idx_ij = sim_ij.max(dim=1)  # [B, Hf, Wf]

        # 取出偏移
        oy = (idx_ij // window_size) - r
        ox = (idx_ij % window_size) - r
        flow_ij = torch.stack([ox, oy], dim=1).float()  # [B, 2, Hf, Wf]

        # j->i（对称）
        Fi_unf = F.unfold(Fi, kernel_size=window_size, padding=r).view(B, C, window_size * window_size, Hf, Wf)
        Fj_e = Fj.unsqueeze(2)
        sim_ji = (Fj_e * Fi_unf).sum(dim=1)  # [B, k*k, Hf, Wf]
        maxv_ji, idx_ji = sim_ji.max(dim=1)
        oy_b = (idx_ji // window_size) - r
        ox_b = (idx_ji % window_size) - r
        flow_ji = torch.stack([ox_b, oy_b], dim=1).float()

        # 互为最近邻：检查正向位移在反向窗口内互相指向
        # 将 i->j 的匹配点坐标，使用 grid_sample 采样反向流，避免高级索引维度错位
        grid_y, grid_x = torch.meshgrid(
            torch.arange(Hf, device=Fi.device, dtype=torch.float32),
            torch.arange(Wf, device=Fi.device, dtype=torch.float32),
            indexing="ij",
        )
        pj_x = grid_x[None] + flow_ij[:, 0]
        pj_y = grid_y[None] + flow_ij[:, 1]
        # 归一化到 [-1,1]
        pj_xn = 2.0 * pj_x / max(Wf - 1, 1) - 1.0
        pj_yn = 2.0 * pj_y / max(Hf - 1, 1) - 1.0
        sample_grid = torch.stack([pj_xn, pj_yn], dim=-1)  # [B,Hf,Wf,2]

        flow_ji_perm = flow_ji  # [B,2,Hf,Wf]
        back_flow = F.grid_sample(
            flow_ji_perm, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B,2,Hf,Wf]

        fb_err = torch.linalg.norm(flow_ij + back_flow, dim=1)  # [B,Hf,Wf]

        mutual = fb_err < 1.0  # 初步互查
        return flow_ij, flow_ji, maxv_ij, maxv_ji, mutual

    @staticmethod
    def _upsample_flow(flow_lr, scale_h: float, scale_w: float, out_hw):
        # 双线性上采样并按比例放大位移
        flow_hr = F.interpolate(flow_lr, size=out_hw, mode="bilinear", align_corners=True)
        flow_hr[:, 0] *= scale_w
        flow_hr[:, 1] *= scale_h
        return flow_hr

    @staticmethod
    def _pose_induced_flow(X_i, T_ij, K):
        """
        根据 3D 点与位姿计算像素位移，返回 (flow, valid_mask, pixel_coords)
        X_i: (B,H,W,3), T_ij: (B,4,4), K: (B,3,3)
        """
        B, H, W, _ = X_i.shape
        ones = torch.ones(B, H * W, 1, device=X_i.device)
        X_flat = X_i.reshape(B, -1, 3)
        X_h = torch.cat([X_flat, ones], dim=-1)  # (B,HW,4)

        X_tr = torch.bmm(T_ij, X_h.transpose(1, 2)).transpose(1, 2)[..., :3]  # (B,HW,3)
        X_pr = torch.bmm(K, X_tr.transpose(1, 2)).transpose(1, 2)  # (B,HW,3)
        pix = X_pr[..., :2] / (X_pr[..., 2:3] + 1e-8)  # (B,HW,2)
        pix = pix.view(B, H, W, 2)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=X_i.device, dtype=torch.float32),
            torch.arange(W, device=X_i.device, dtype=torch.float32),
            indexing="ij",
        )
        orig = torch.stack([grid_x, grid_y], dim=-1)[None].expand(B, -1, -1, -1)
        flow = (pix - orig).permute(0, 3, 1, 2)

        valid = (
            (pix[..., 0] >= 0)
            & (pix[..., 0] <= W - 1)
            & (pix[..., 1] >= 0)
            & (pix[..., 1] <= H - 1)
            & (X_tr[..., 2].view(B, H, W) > 0)
        )
        return flow, valid, pix

    def compute_loss(self, gts, preds, **kw):
        if len(gts) < 2:
            return torch.tensor(0.0, device=preds[0]["pts3d_in_self_view"].device), {}

        B, C, H, W = gts[0]["img"].shape
        total, n = 0.0, 0
        details = {}

        for i in range(len(gts) - 1):
            img_i = gts[i]["img"]
            img_j = gts[i + 1]["img"]

            # 1) DINOv2 特征
            with torch.no_grad():
                Fi = self.extractor(img_i)
                Fj = self.extractor(img_j)

            # 2) 局部互NN伪光流（特征分辨率）
            flow_ij_lr, flow_ji_lr, sim_ij, sim_ji, mutual = self._mutual_nn_local_flow(
                Fi, Fj, self.window_size
            )

            Hf, Wf = Fi.shape[-2:]
            scale_h, scale_w = H / Hf, W / Wf
            # 相似度与互查掩码
            sim_mask = (sim_ij > self.sim_threshold) & mutual

            # 3) 上采样伪光流到原图分辨率
            flow_ij = self._upsample_flow(flow_ij_lr, scale_h, scale_w, (H, W))

            # 4) 位姿诱导光流（对深度 stop-grad）
            X_i = preds[i]["pts3d_in_self_view"]
            if self.stop_grad_depth:
                X_i = X_i.detach()

            if ("camera_intrinsics" not in gts[i] and "camera_intrinsics" not in gts[i+1]) or \
               ("camera_pose" not in preds[i] or "camera_pose" not in preds[i + 1]):
                continue

            # 计算相对位姿：从第 i 相机坐标到第 j 相机坐标
            C_i = pose_encoding_to_camera(preds[i]["camera_pose"])      # camera-to-world
            C_j = pose_encoding_to_camera(preds[i + 1]["camera_pose"])  # camera-to-world
            T_ij = torch.bmm(inv(C_j), C_i)  # i(cam) -> j(cam)

            # 使用第 j 帧的内参做投影
            K = gts[i + 1].get("camera_intrinsics", gts[i]["camera_intrinsics"])  # (B,3,3)
            flow_pose, valid_pose, _ = self._pose_induced_flow(X_i, T_ij, K)

            # 5) 有效掩码：可见性 + 相似度 + 前后向一致性（在特征分辨率做过，放大对齐）
            sim_mask_hr = sim_mask.float().unsqueeze(1)
            sim_mask_hr = F.interpolate(sim_mask_hr, size=(H, W), mode="nearest").squeeze(1).bool()
            valid = valid_pose & sim_mask_hr

            if valid.sum() == 0:
                continue

            # 6) L1 一致性损失
            diff = (flow_pose - flow_ij).permute(0, 2, 3, 1)[valid]
            loss_ij = diff.abs().mean()

            total = total + loss_ij
            n += 1
            details[f"dino_flow_loss_{i}_{i+1}"] = float(loss_ij)

            if i == 0:
                details["dino_flow_vis_flow_pose"] = flow_pose.detach()
                details["dino_flow_vis_flow_dino"] = flow_ij.detach()

        if n == 0:
            return torch.tensor(0.0, device=preds[0]["pts3d_in_self_view"].device), details

        return total / n, details


class GeometricConsistencyLoss(MultiLoss):
    """
    纯几何一致性自监督损失 - 基于时序重投影一致性
    
    核心思想：
    如果模型的预测是几何自洽的，那么：
    pred_depth_t 应该等于 warp(pred_depth_{t-1}, pred_relative_pose_t)
    
    实现步骤：
    1. 用 pred_depth_{t-1}.detach() 构建 t-1 时刻的 3D 点云（固定历史）
    2. 用 pred_relative_pose_t 将点云变换到 t 坐标系
    3. 重投影得到 warped_depth_t
    4. 计算 L_geo = robust_loss(pred_depth_t - warped_depth_t, M_valid)
    
    关键设计：
    - detach() 打破对称性，强制优化当前预测而非历史篡改
    - 三重掩码确保只在物理合理区域计算
    - 对光照变化免疫，纯几何约束
    """
    
    def __init__(self, 
                 robust_loss: bool = True, 
                 consistency_thresh: float = 1.0,
                 depth_thresh: float = 0.01,
                 loss_norm: str = "mean"):
        super().__init__()
        self.robust_loss = robust_loss  # 使用 Charbonnier 而非 L2
        self.consistency_thresh = consistency_thresh  # 前后向一致性阈值（像素）
        self.depth_thresh = depth_thresh  # 深度有效性阈值
        self.loss_norm = loss_norm  # "mean" 或 "sum"
    
    def get_name(self):
        return "GeometricConsistencyLoss"
    
    @staticmethod
    def backproject_depth(depth, intrinsics):
        """
        深度图反投影为 3D 点云
        
        参数:
            depth: (B, H, W) 深度图
            intrinsics: (B, 3, 3) 相机内参矩阵
            
        返回:
            points_3d: (B, H, W, 3) 3D 点云，在相机坐标系下
        """
        B, H, W = depth.shape
        device = depth.device
        
        # 构建像素网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 齐次像素坐标 (H, W, 3)
        pixel_coords = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
        pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)
        
        # 反投影：P_cam = depth * K^{-1} * [u, v, 1]^T
        # intrinsics_inv: (B, 3, 3)
        intrinsics_inv = torch.inverse(intrinsics)  # (B, 3, 3)
        
        # 广播计算：(B, H, W, 3) @ (B, 3, 3)^T -> (B, H, W, 3)
        # 重塑为 (B*H*W, 3) 做矩阵乘法
        pixel_coords_flat = pixel_coords.reshape(B, -1, 3)  # (B, H*W, 3)
        points_cam_flat = torch.bmm(pixel_coords_flat, intrinsics_inv.transpose(-1, -2))  # (B, H*W, 3)
        points_cam = points_cam_flat.reshape(B, H, W, 3)  # (B, H, W, 3)
        
        # 乘以深度
        depth_expanded = depth.unsqueeze(-1)  # (B, H, W, 1)
        points_3d = points_cam * depth_expanded  # (B, H, W, 3)
        
        return points_3d
    
    @staticmethod
    def transform_points(points_3d, pose_matrix):
        """
        用位姿矩阵变换 3D 点云
        
        参数:
            points_3d: (B, H, W, 3) 3D 点云
            pose_matrix: (B, 4, 4) 位姿变换矩阵
            
        返回:
            transformed_points: (B, H, W, 3) 变换后的 3D 点云
        """
        B, H, W, _ = points_3d.shape
        
        # 转换为齐次坐标 (B, H*W, 4)
        points_flat = points_3d.reshape(B, -1, 3)  # (B, H*W, 3)
        ones = torch.ones(B, H*W, 1, device=points_3d.device)
        points_homo = torch.cat([points_flat, ones], dim=-1)  # (B, H*W, 4)
        
        # 应用变换：(B, 4, 4) @ (B, H*W, 4)^T -> (B, 4, H*W)
        transformed_homo = torch.bmm(pose_matrix, points_homo.transpose(-1, -2))  # (B, 4, H*W)
        transformed_points = transformed_homo[:, :3, :].transpose(-1, -2)  # (B, H*W, 3)
        
        return transformed_points.reshape(B, H, W, 3)  # (B, H, W, 3)
    
    @staticmethod
    def project_points(points_3d, intrinsics):
        """
        3D 点云投影到 2D 像素平面
        
        参数:
            points_3d: (B, H, W, 3) 3D 点云
            intrinsics: (B, 3, 3) 相机内参矩阵
            
        返回:
            pixel_coords: (B, H, W, 2) 像素坐标
            depths: (B, H, W) 投影后的深度值
            valid_mask: (B, H, W) 有效性掩码（正深度且在画面内）
        """
        B, H, W, _ = points_3d.shape
        
        # 重塑并投影：K @ P_cam
        points_flat = points_3d.reshape(B, -1, 3)  # (B, H*W, 3)
        projected_flat = torch.bmm(intrinsics, points_flat.transpose(-1, -2))  # (B, 3, H*W)
        projected = projected_flat.transpose(-1, -2).reshape(B, H, W, 3)  # (B, H, W, 3)
        
        # 透视除法
        depths = projected[..., 2]  # (B, H, W)
        pixel_coords = projected[..., :2] / (depths.unsqueeze(-1) + 1e-8)  # (B, H, W, 2)
        
        # 构建有效性掩码
        pos_depth_mask = depths > 0  # 正深度
        in_frame_mask = (
            (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < W) &
            (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < H)
        )  # 画面内
        valid_mask = pos_depth_mask & in_frame_mask
        
        return pixel_coords, depths, valid_mask
    
    def compute_forward_backward_consistency(self, flow_forward, flow_backward, thresh=1.0):
        """
        计算前后向光流一致性掩码
        
        参数:
            flow_forward: (B, H, W, 2) 前向光流 t-1 -> t
            flow_backward: (B, H, W, 2) 后向光流 t -> t-1
            thresh: 一致性阈值（像素）
            
        返回:
            consistency_mask: (B, H, W) 一致性掩码
        """
        B, H, W, _ = flow_forward.shape
        device = flow_forward.device
        
        # 构建像素网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # 前向映射：t-1 -> t
        target_coords = grid + flow_forward  # (B, H, W, 2)
        
        # 归一化坐标用于 grid_sample
        target_coords_norm = torch.zeros_like(target_coords)
        target_coords_norm[..., 0] = 2.0 * target_coords[..., 0] / max(W - 1, 1) - 1.0
        target_coords_norm[..., 1] = 2.0 * target_coords[..., 1] / max(H - 1, 1) - 1.0
        
        # 在目标位置采样后向光流
        flow_backward_perm = flow_backward.permute(0, 3, 1, 2)  # (B, 2, H, W)
        sampled_backward_flow = F.grid_sample(
            flow_backward_perm, target_coords_norm,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        # 计算一致性误差：||flow_forward + sampled_backward_flow||
        consistency_error = torch.norm(flow_forward + sampled_backward_flow, dim=-1)  # (B, H, W)
        
        # 一致性掩码
        consistency_mask = consistency_error < thresh
        
        return consistency_mask
    
    def compute_loss(self, gts, preds, **kw):
        """
        计算几何一致性损失
        
        参数:
            gts: 真实数据列表，包含相机内参
            preds: 预测数据列表，包含深度和位姿
            
        返回:
            (geo_loss, details): 几何一致性损失和详细信息
        """
        if len(gts) < 2 or len(preds) < 2:
            # 需要至少两帧进行几何一致性计算
            return torch.tensor(0.0, device=preds[0]["pts3d_in_self_view"].device), {}
        
        total_loss = 0.0
        valid_pairs = 0
        details = {}
        
        # 处理连续帧对 t-1 -> t
        for i in range(len(gts) - 1):
            try:
                # 检查必要数据的可用性
                if "camera_intrinsics" not in gts[i] or "camera_intrinsics" not in gts[i+1]:
                    continue
                if "camera_pose" not in preds[i] or "camera_pose" not in preds[i+1]:
                    continue
                
                # 提取数据
                # 深度：使用 pts3d_in_self_view 的 z 分量
                depth_t_minus_1 = preds[i]["pts3d_in_self_view"][..., 2]  # (B, H, W)
                depth_t = preds[i+1]["pts3d_in_self_view"][..., 2]  # (B, H, W)
                
                # 相机内参
                K_t_minus_1 = gts[i]["camera_intrinsics"]  # (B, 3, 3)
                K_t = gts[i+1]["camera_intrinsics"]  # (B, 3, 3)
                
                # 相机位姿（camera-to-world）
                pose_t_minus_1 = pose_encoding_to_camera(preds[i]["camera_pose"])  # (B, 4, 4)
                pose_t = pose_encoding_to_camera(preds[i+1]["camera_pose"])  # (B, 4, 4)
                
                # 计算相对位姿：t-1 -> t （world坐标系下：先逆变换到world，再变换到t）
                # T_{t-1->t} = T_t^{-1} @ T_{t-1}
                relative_pose = torch.bmm(inv(pose_t), pose_t_minus_1)  # (B, 4, 4)
                
                # Step 1: 构建 t-1 时刻的 3D 点云（detach 防止梯度回流）
                depth_t_minus_1_detached = depth_t_minus_1.detach()
                points_3d_t_minus_1 = self.backproject_depth(depth_t_minus_1_detached, K_t_minus_1)  # (B, H, W, 3)
                
                # Step 2: 用相对位姿变换到 t 时刻坐标系
                points_3d_warped = self.transform_points(points_3d_t_minus_1, relative_pose)  # (B, H, W, 3)
                
                # Step 3: 重投影到 t 时刻图像平面
                pixel_coords_warped, depth_warped, projection_valid = self.project_points(points_3d_warped, K_t)
                
                # Step 4: 构建三重有效性掩码
                
                # 掩码 1: 投影有效性（已在 project_points 中计算）
                M_projection = projection_valid
                
                # 掩码 2: 深度有效性
                M_depth = (depth_t > self.depth_thresh) & (depth_warped > self.depth_thresh)
                
                # 掩码 3: 前后向一致性（可选，需要计算反向流）
                # 为简化首次实现，暂时使用投影和深度掩码
                # TODO: 后续可添加完整的前后向一致性检查
                
                # 组合掩码
                M_valid = M_projection & M_depth  # (B, H, W)
                
                if M_valid.sum() == 0:
                    continue
                
                # Step 5: 计算几何一致性损失
                depth_diff = depth_t - depth_warped  # (B, H, W)
                
                if self.robust_loss:
                    # Charbonnier 损失：sqrt(x^2 + eps^2) - eps
                    eps = 0.001
                    pixel_losses = torch.sqrt(depth_diff**2 + eps**2) - eps
                else:
                    # L1 损失
                    pixel_losses = torch.abs(depth_diff)
                
                # 只在有效区域计算损失
                valid_losses = pixel_losses[M_valid]
                
                if self.loss_norm == "mean":
                    pair_loss = valid_losses.mean()
                else:  # "sum"
                    pair_loss = valid_losses.sum() / M_valid.sum().clamp(min=1)
                
                total_loss += pair_loss
                valid_pairs += 1
                
                # 记录详细信息
                details[f"geo_loss_{i}_{i+1}"] = float(pair_loss)
                details[f"geo_valid_pixels_{i}_{i+1}"] = int(M_valid.sum())
                details[f"geo_valid_ratio_{i}_{i+1}"] = float(M_valid.float().mean())
                
                # 记录第一对的可视化信息
                if i == 0:
                    details["geo_depth_diff"] = depth_diff.detach()
                    details["geo_valid_mask"] = M_valid.detach()
                    details["geo_warped_depth"] = depth_warped.detach()
                    details["geo_current_depth"] = depth_t.detach()
                
            except Exception as e:
                # 如果某一对处理失败，记录但继续处理其他对
                print(f"Warning: Failed to compute geometric consistency loss for pair {i}-{i+1}: {e}")
                continue
        
        # 计算平均损失
        if valid_pairs > 0:
            avg_loss = total_loss / valid_pairs
            details["geo_valid_pairs"] = valid_pairs
        else:
            avg_loss = torch.tensor(0.0, device=preds[0]["pts3d_in_self_view"].device)
            details["geo_valid_pairs"] = 0
        
        return avg_loss, details


class DecoupledTotalLoss(MultiLoss):
    """
    解耦的总损失函数

    实现新的损失组合策略：
    L_total = L_3D_conf + w_pose * L_pose + w_rgb * L_rgb + w_dflow * L_dflow

    核心特点：
    1. 独立控制每个损失项的权重
    2. 支持动态权重调度
    3. 详细的损失监控和梯度分析
    4. 解决位姿学习的"偷懒"问题

    使用场景：
    - 需要精确控制不同损失项权重的训练
    - 解决弱耦合问题的训练策略
    - 复杂场景的多约束优化
    """

    def __init__(
        self,
        conf_loss,                    # ConfLoss_Decoupled实例
        rgb_loss,                     # RGBLoss实例
        dflow_loss,                   # DynamicsAwareFlowLoss实例
        weight_scheduler=None,        # WeightScheduler实例
        w_pose=0.1,                   # 位姿损失权重（默认较低）
        w_rgb=1.0,                    # RGB损失权重
        w_dflow=2.0,                  # 光流损失权重（默认较高）
        monitor_gradients=False,      # 是否监控梯度范数
        current_epoch=0,              # 当前训练轮次
        total_epochs=100              # 总训练轮次
    ):
        super().__init__()

        self.conf_loss = conf_loss
        self.rgb_loss = rgb_loss
        self.dflow_loss = dflow_loss
        self.weight_scheduler = weight_scheduler

        # 权重设置
        self.w_pose = w_pose
        self.w_rgb = w_rgb
        self.w_dflow = w_dflow

        # 监控设置
        self.monitor_gradients = monitor_gradients
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

    def get_name(self):
        """获取损失函数名称"""
        return "DecoupledTotalLoss"

    def update_epoch(self, epoch):
        """更新当前训练轮次，用于权重调度"""
        self.current_epoch = epoch

    def get_current_weights(self):
        """获取当前的损失权重"""
        if self.weight_scheduler is not None:
            # 使用权重调度器
            weights = self.weight_scheduler.get_weights(self.current_epoch, self.total_epochs)
            return {
                'w_pose': weights.get('w_pose', self.w_pose),
                'w_rgb': weights.get('w_rgb', self.w_rgb),
                'w_dflow': weights.get('w_dflow', self.w_dflow)
            }
        else:
            # 使用固定权重
            return {
                'w_pose': self.w_pose,
                'w_rgb': self.w_rgb,
                'w_dflow': self.w_dflow
            }

    def compute_gradient_norms(self, losses, model_parameters):
        """
        计算各损失项的梯度范数

        用于监控不同损失项对模型参数的影响程度
        """
        grad_norms = {}

        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor) and loss_value.requires_grad:
                # 计算当前损失对模型参数的梯度
                grads = torch.autograd.grad(
                    loss_value, model_parameters,
                    retain_graph=True, create_graph=False, allow_unused=True
                )

                # 计算梯度范数
                grad_norm = 0.0
                for grad in grads:
                    if grad is not None:
                        grad_norm += grad.norm().item() ** 2
                grad_norms[f"grad_norm_{loss_name}"] = grad_norm ** 0.5

        return grad_norms

    def compute_loss(self, gts, preds, **kw):
        """
        计算解耦的总损失

        参数:
            gts: 真实数据列表
            preds: 预测数据列表
            **kw: 其他关键字参数

        返回:
            (total_loss, details): 总损失和详细信息
        """
        # 获取当前权重
        current_weights = self.get_current_weights()

        # 1. 计算各项损失
        conf_loss, conf_details = self.conf_loss(gts, preds, **kw)
        rgb_loss, rgb_details = self.rgb_loss(gts, preds, **kw)
        dflow_loss, dflow_details = self.dflow_loss(gts, preds, **kw)

        # 2. 提取pose_loss（从conf_details中）
        pose_loss = conf_details.get("pose_loss", torch.tensor(0.0, device=conf_loss.device))
        scale_loss = conf_details.get("scale_loss", torch.tensor(0.0, device=conf_loss.device))

        # 3. 计算加权损失
        weighted_conf_loss = conf_loss  # 3D置信度损失权重固定为1.0
        weighted_pose_loss = current_weights['w_pose'] * pose_loss
        weighted_rgb_loss = current_weights['w_rgb'] * rgb_loss
        weighted_dflow_loss = current_weights['w_dflow'] * dflow_loss
        weighted_scale_loss = scale_loss  # 尺度损失权重固定为1.0

        # 4. 组合总损失
        total_loss = (weighted_conf_loss +
                     weighted_pose_loss +
                     weighted_rgb_loss +
                     weighted_dflow_loss +
                     weighted_scale_loss)

        # 5. 合并详细信息
        details = {}
        details.update(conf_details)
        details.update(rgb_details)
        details.update(dflow_details)

        # 6. 添加权重和损失分解信息
        details.update({
            # 当前权重
            "current_w_pose": current_weights['w_pose'],
            "current_w_rgb": current_weights['w_rgb'],
            "current_w_dflow": current_weights['w_dflow'],

            # 原始损失值
            "raw_conf_loss": float(conf_loss),
            "raw_pose_loss": float(pose_loss),
            "raw_rgb_loss": float(rgb_loss),
            "raw_dflow_loss": float(dflow_loss),
            "raw_scale_loss": float(scale_loss),

            # 加权损失值
            "weighted_conf_loss": float(weighted_conf_loss),
            "weighted_pose_loss": float(weighted_pose_loss),
            "weighted_rgb_loss": float(weighted_rgb_loss),
            "weighted_dflow_loss": float(weighted_dflow_loss),
            "weighted_scale_loss": float(weighted_scale_loss),

            # 总损失
            "total_loss": float(total_loss),

            # 训练信息
            "current_epoch": self.current_epoch
        })

        # 7. 梯度监控（可选）
        if self.monitor_gradients and hasattr(kw, 'model_parameters'):
            individual_losses = {
                'conf': weighted_conf_loss,
                'pose': weighted_pose_loss,
                'rgb': weighted_rgb_loss,
                'dflow': weighted_dflow_loss,
                'scale': weighted_scale_loss
            }
            grad_norms = self.compute_gradient_norms(individual_losses, kw['model_parameters'])
            details.update(grad_norms)

        return total_loss, details





# ==============================================================================
#                      (请将此代码块放在您的 losses.py 文件中)
# ==============================================================================
import sys
import os

# 检查RAFT是否可用 (请将此段代码放在文件顶部，确保import sys, os也在顶部)
try:
    # 为了从您clone的仓库加载RAFT，我们需要将其路径添加到Python的搜索路径中
    RAFT_REPO_PATH = "/hy-tmp/hy-tmp/CUT3R/RAFT"  # 这是您clone的RAFT仓库的根目录
    if RAFT_REPO_PATH not in sys.path:
        sys.path.append(RAFT_REPO_PATH)
    
    from raft import RAFT
    from utils.utils import InputPadder # 这是RAFT仓库自带的工具
    RAFT_AVAILABLE = True
    print(f"✅ RAFT found and imported from local repository: {RAFT_REPO_PATH}")

except ImportError as e:
    RAFT_AVAILABLE = False
    print(f"⚠️ Warning: Could not import RAFT from local repository. Error: {e}")
    print("DynamicsAwareFlowLoss will be disabled.")


class DynamicsAwareFlowLoss(MultiLoss):
    """
    根据Endo3R论文修正后的动态光流损失实现。
    此版本修正了梯度流和坐标系变换问题，并从本地加载RAFT模型。
    """
    def __init__(self, raft_weights_path="/hy-tmp/hy-tmp/CUT3R/RAFT/models/raft_pytorch_large.pth"):
        super().__init__()
        self.raft_model = None
        if RAFT_AVAILABLE:
            self._init_raft_model(raft_weights_path)

    def _init_raft_model(self, raft_weights_path):
        """从您指定的本地路径初始化并冻结RAFT模型。"""
        print("🔄 Initializing RAFT model from local path...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # RAFT模型需要一个参数字典来进行初始化
            import argparse
            args = argparse.Namespace()
            args.small = False
            args.mixed_precision = False # 如果您使用FP16训练，可以设为True
            
            self.raft_model = RAFT(args)
            
            print(f"📁 Loading RAFT weights from: {raft_weights_path}")
            # RAFT的权重保存在一个DataParallel封装的模型中，需要移除'module.'前缀
            pretrained_weights = torch.load(raft_weights_path, map_location=device)
            self.raft_model.load_state_dict(pretrained_weights)
            
            self.raft_model = self.raft_model.to(device)
            self.raft_model.eval()
            
            # 冻结所有参数
            for param in self.raft_model.parameters():
                param.requires_grad = False
            
            print("✅ RAFT model initialized successfully from local weights.")

        except Exception as e:
            print(f"❌ FATAL: Could not initialize RAFT from local files: {e}")
            self.raft_model = None


    def get_name(self):
        return "DynamicsAwareFlowLoss"

    @torch.no_grad()
    def compute_pseudo_gt_flow(self, img1, img2):
        """使用RAFT计算光流，作为伪真值。"""
        if self.raft_model is None: return None

        # RAFT仓库的推理方式通常需要对输入进行填充 (padding)
        padder = InputPadder(img1.shape)
        # RAFT期望的输入范围是 [0, 255]
        img1_padded, img2_padded = padder.pad(img1*255.0, img2*255.0)

        # RAFT返回两个值，我们只需要光流预测
        _, flow_up = self.raft_model(img1_padded, img2_padded, iters=20, test_mode=True)
        
        # 移除padding，得到原始尺寸的光流
        return padder.unpad(flow_up)

    def compute_loss(self, gts, preds, **kw):
        if len(gts) < 2 or not RAFT_AVAILABLE or self.raft_model is None:
            device = preds[0]["pts3d_in_self_view"].device
            return torch.tensor(0.0, device=device), {}

        total_loss = 0.0
        valid_pairs = 0
        details = {}
        device = preds[0]["pts3d_in_self_view"].device

        for i in range(len(gts) - 1):
            j = i + 1

            # --- 0. 数据准备 ---
            img_i, img_j = gts[i]["img"], gts[j]["img"]
            X_i_pred, X_j_pred = preds[i]["pts3d_in_self_view"], preds[j]["pts3d_in_self_view"]
            T_i_pred = pose_encoding_to_camera(preds[i]["camera_pose"])
            T_j_pred = pose_encoding_to_camera(preds[j]["camera_pose"])
            K_pred = gts[j].get("camera_intrinsics", gts[i].get("camera_intrinsics"))

            if K_pred is None: continue

            B, _, H, W = img_i.shape
            
            # --- 1. 计算伪真值光流 (O_i->j) ---
            O_i_to_j = self.compute_pseudo_gt_flow(img_i, img_j)
            if O_i_to_j is None: continue

            # --- 2. 计算场景流 (S_i->j) ---
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1).float().unsqueeze(0) # (1, H, W, 2)
            
            flow_permuted = O_i_to_j.permute(0, 2, 3, 1) # (B, H, W, 2)
            coords_in_j = grid + flow_permuted

            coords_in_j_norm = coords_in_j.clone()
            coords_in_j_norm[..., 0] = 2.0 * coords_in_j[..., 0] / (W - 1) - 1.0
            coords_in_j_norm[..., 1] = 2.0 * coords_in_j[..., 1] / (H - 1) - 1.0

            X_j_warped = F.grid_sample(X_j_pred.permute(0, 3, 1, 2), coords_in_j_norm, 
                                       mode='bilinear', padding_mode='border', align_corners=True).permute(0, 2, 3, 1)

            # ✅ 关键修正 #1: 梯度必须通过场景流回传到点云预测。
            S_i_to_j = X_j_warped - X_i_pred

            # --- 3. 计算估计光流 (f_hat_i->j) ---
            X_i_deformed = X_i_pred + S_i_to_j

            # ✅ 关键修正 #2: 使用正确的相对位姿 T_{i->j} 来变换点云。
            T_i_to_j = torch.bmm(inv(T_j_pred), T_i_pred)
            
            X_i_deformed_flat = X_i_deformed.view(B, -1, 3)
            ones = torch.ones(B, H*W, 1, device=device)
            X_i_deformed_homo = torch.cat([X_i_deformed_flat, ones], dim=-1)
            X_in_j_homo = torch.bmm(T_i_to_j, X_i_deformed_homo.transpose(1, 2))
            X_in_j = X_in_j_homo[:, :3, :].transpose(1, 2)

            projected_points = torch.bmm(K_pred, X_in_j.transpose(1, 2)).transpose(1, 2)
            
            z = projected_points[..., 2:3].clamp(min=1e-6)
            pixel_coords_j = (projected_points[..., :2] / z).view(B, H, W, 2)

            f_hat_i_to_j = (pixel_coords_j - grid).permute(0, 3, 1, 2)

            # --- 4. 计算损失 ---
            valid_flow_mask = (coords_in_j[..., 0] >= 0) & (coords_in_j[..., 0] < W) & \
                              (coords_in_j[..., 1] >= 0) & (coords_in_j[..., 1] < H)
            valid_depth_mask = (z.view(B, H, W) > 1e-3)
            final_valid_mask = (valid_flow_mask & valid_depth_mask).unsqueeze(1).detach()

            flow_diff = f_hat_i_to_j - O_i_to_j
            
            if final_valid_mask.sum() > 0:
                loss = torch.abs(flow_diff[final_valid_mask.expand_as(flow_diff)]).mean()
                total_loss = total_loss + loss
                valid_pairs += 1
                details[f"dflow_loss_{i}_{j}"] = float(loss)

        if valid_pairs > 0:
            avg_loss = total_loss / valid_pairs
        else:
            avg_loss = torch.tensor(0.0, device=device)

        details['dflow_avg_loss'] = float(avg_loss)
        return avg_loss, details


class SmoothnessLoss(MultiLoss):
    def __init__(self, detach_depth=True):
        super().__init__()
        self.detach_depth = detach_depth

    def get_name(self):
        return "SmoothnessLoss"

    def compute_loss(self, gts, preds, **kw):
        # 1. Get predicted depth and image from the first view (anchor view)
        depth = preds[0]["pts3d_in_self_view"][..., 2]  # Shape (B, H, W)
        if self.detach_depth:
            depth = depth.detach()

        image = gts[0]["img"]  # Shape (B, C, H, W)

        # 2. Convert depth to disparity and normalize
        disp = 1.0 / (depth + 1e-7)
        mean_disp = disp.mean(1, True).mean(2, True)
        norm_disp = disp / (mean_disp + 1e-7)

        # Add channel dimension to disparity map for get_smooth_loss
        norm_disp = norm_disp.unsqueeze(1)  # Shape (B, 1, H, W)

        # 3. Compute smoothness loss
        smooth_loss = get_smooth_loss(norm_disp, image)

        details = {self.get_name(): float(smooth_loss)}
        return smooth_loss, details


# ==============================================================================
#                           Endo3R Dynamic-Aware Flow Loss
# ==============================================================================

class Endo3RFlowLoss(MultiLoss):
    """
    Endo3R动态感知光流损失实现
    
    核心原理：
    1. 将观察光流分解为：相机运动引起的光流 + 场景自身运动的光流
    2. 通过光流一致性约束来自监督训练深度和位姿估计
    3. 避免依赖不准确的GT深度监督，解决监督信号冲突问题
    
    参考论文：Endo3R
    公式4: S_{i→j}(u) = X_{j,i}(u + O_{i→j}(u)) - X_{i,i}(u)
    公式5: f̂_{i→j}(u') = KT_{j,i}(X_{i,i}(u') + S_{i→j}(u')) - u'
    公式6: L^{i→j}_{Dflow} = ||f̂_{i→j}(u') - O_{i→j}(u')||_1
    """
    
    def __init__(self, raft_model_path=None, flow_strategy="anchor_based"):
        """
        Args:
            raft_model_path: RAFT权重路径
            flow_strategy: 光流计算策略
                - "anchor_based": 所有视角与第一视角计算 (与ConfLoss一致)
                - "consecutive": 连续帧对计算 (原实现)
                - "mixed": 混合策略，主要用anchor，辅助用consecutive
        """
        super().__init__()
        self.raft_model = None
        self.flow_strategy = flow_strategy
        
        # 修复RAFT路径配置 - 使用用户确认的可用路径
        if raft_model_path is None:
            # 使用用户确认可用的RAFT权重路径
            raft_model_path = "/hy-tmp/hy-tmp/monst3r/third_party/RAFT/models/raft-things.pth"
        
        self.raft_model_path = raft_model_path
        self._load_raft_model()
    
    def _load_raft_model(self):
        """预加载RAFT模型，避免重复加载"""
        try:
            import sys
            import os
            # 添加RAFT路径到sys.path - 修复路径配置
            possible_raft_paths = [
                "/hy-tmp/hy-tmp/CUT3R/RAFT",
                "/hy-tmp/monst3r/third_party/RAFT",
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../RAFT"))
            ]
            
            raft_path = None
            for path in possible_raft_paths:
                if os.path.exists(os.path.join(path, "core", "raft.py")):
                    raft_path = path
                    break
            
            if raft_path is None:
                raise ImportError("无法找到RAFT代码路径")
                
            if raft_path not in sys.path:
                sys.path.insert(0, raft_path)
            
            # 添加RAFT core路径以解决相对导入问题
            raft_core_path = os.path.join(raft_path, "core")
            if raft_core_path not in sys.path:
                sys.path.insert(0, raft_core_path)
            
            # 导入RAFT
            from core.raft import RAFT
            import argparse
            
            # 创建RAFT模型参数
            args = argparse.Namespace()
            args.small = False
            args.mixed_precision = False
            args.alternate_corr = False
            
            # 初始化模型
            self.raft_model = RAFT(args)
            
            # 加载预训练权重（如果存在）
            # 如果是绝对路径，直接使用；否则转为绝对路径
            if os.path.isabs(self.raft_model_path):
                model_path = self.raft_model_path
            else:
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.raft_model_path))
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # 处理DataParallel保存的权重（去除'module.'前缀）
                if 'module.fnet.conv1.weight' in checkpoint:
                    print("检测到DataParallel权重，去除'module.'前缀...")
                    new_checkpoint = {}
                    for key, value in checkpoint.items():
                        if key.startswith('module.'):
                            new_key = key[7:]  # 去除'module.'前缀
                            new_checkpoint[new_key] = value
                        else:
                            new_checkpoint[key] = value
                    checkpoint = new_checkpoint
                
                self.raft_model.load_state_dict(checkpoint)
                print(f"✅ RAFT模型权重已加载: {model_path}")
            else:
                print(f"⚠️ RAFT权重文件未找到: {model_path}，使用随机初始化权重")
            
            self.raft_model.eval()
            # 冻结RAFT参数
            for param in self.raft_model.parameters():
                param.requires_grad = False
                
            print("✅ RAFT模型初始化完成")
            
        except Exception as e:
            print(f"❌ RAFT模型加载失败: {e}")
            print("   将禁用Endo3R光流损失")
            self.raft_model = None
    
    def get_name(self):
        return "Endo3RFlowLoss"
    
    def _estimate_intrinsics_from_pointcloud(self, pts3d):
        """
        使用CUT3R的方法从点云估计内参矩阵
        参考：demo_online.py 第255-257行和post_process.py的estimate_focal_knowing_depth函数
        """
        from dust3r.post_process import estimate_focal_knowing_depth
        
        B, H, W, _ = pts3d.shape
        device = pts3d.device
        
        # 主点假设在图像中心（与CUT3R一致）
        pp = torch.tensor([W // 2, H // 2], device=device).float().repeat(B, 1)
        
        # 使用CUT3R的weiszfeld算法估计焦距
        focal = estimate_focal_knowing_depth(pts3d, pp, focal_mode="weiszfeld")
        
        # 构建内参矩阵
        K = torch.zeros(B, 3, 3, device=device)
        K[:, 0, 0] = focal  # fx
        K[:, 1, 1] = focal  # fy (假设fx=fy)
        K[:, 0, 2] = pp[:, 0]  # cx
        K[:, 1, 2] = pp[:, 1]  # cy
        K[:, 2, 2] = 1.0
        
        return K
    
    @torch.no_grad()
    def _compute_observed_flow(self, img1, img2):
        """
        使用RAFT计算观察光流 O_{i→j}
        
        Args:
            img1: 帧i的图像 (B, H, W, 3), 范围[0,1]
            img2: 帧j的图像 (B, H, W, 3), 范围[0,1]
            
        Returns:
            flow: 光流 (B, 2, H, W), 单位：像素
        """
        if self.raft_model is None:
            return None
        
        device = img1.device
        self.raft_model = self.raft_model.to(device)
        
        # 调试：检查输入格式
        print(f"Debug RAFT input - img1 shape: {img1.shape}, img2 shape: {img2.shape}")
        print(f"Debug RAFT input - img1 dtype: {img1.dtype}, range: [{img1.min():.3f}, {img1.max():.3f}]")
        
        # 检查输入格式并转换
        if len(img1.shape) == 4:
            if img1.shape[1] == 3:  # (B, 3, H, W) 格式
                img1_raft = img1 * 255
                img2_raft = img2 * 255
            elif img1.shape[3] == 3:  # (B, H, W, 3) 格式
                img1_raft = img1.permute(0, 3, 1, 2) * 255  # (B, 3, H, W)
                img2_raft = img2.permute(0, 3, 1, 2) * 255
            else:
                print(f"❌ 不支持的图像格式: {img1.shape}, 期望 (B,3,H,W) 或 (B,H,W,3)")
                return None
        else:
            print(f"❌ 不支持的图像维度: {img1.shape}, 期望4D tensor")
            return None
        
        print(f"Debug RAFT converted - img1_raft shape: {img1_raft.shape}, range: [{img1_raft.min():.1f}, {img1_raft.max():.1f}]")
        
        try:
            # RAFT推理
            _, flow = self.raft_model(img1_raft, img2_raft, iters=20, test_mode=True)
            
            # 检查输出有效性
            if torch.isnan(flow).any() or torch.isinf(flow).any():
                print("⚠️ RAFT输出包含NaN或Inf，使用零光流替代")
                # 对于异常情况，返回零光流而不是None，这样训练可以继续
                flow = torch.zeros_like(flow)
                return flow
                
            return flow  # (B, 2, H, W)
        except Exception as e:
            print(f"RAFT计算失败: {e}")
            return None
    
    def _compute_scene_flow(self, X_i, X_j, observed_flow):
        """
        计算场景光流 S_{i→j} (Endo3R公式4)
        S_{i→j}(u) = X_{j,i}(u + O_{i→j}(u)) - X_{i,i}(u)
        
        物理含义：3D点在世界坐标系中的真实移动向量
        
        Args:
            X_i: 帧i的3D点云 (B, H, W, 3)
            X_j: 帧j的3D点云 (B, H, W, 3) 
            observed_flow: 观察光流 (B, 2, H, W)
            
        Returns:
            scene_flow: 场景光流 (B, H, W, 3)
        """
        B, H, W, _ = X_i.shape
        device = X_i.device
        
        # 构建像素网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        
        # 计算对应像素位置: u + O_{i→j}(u)
        flow_permuted = observed_flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        corresponding_coords = grid + flow_permuted  # (B, H, W, 2)
        
        # 归一化坐标到[-1, 1]范围用于grid_sample
        corresponding_coords_norm = corresponding_coords.clone()
        corresponding_coords_norm[..., 0] = 2.0 * corresponding_coords[..., 0] / (W - 1) - 1.0
        corresponding_coords_norm[..., 1] = 2.0 * corresponding_coords[..., 1] / (H - 1) - 1.0
        
        # 在X_j中采样对应点的3D位置: X_{j,i}(u + O_{i→j}(u))
        # 注意：这里需要保持梯度，因为要优化X_j
        X_j_sampled = F.grid_sample(
            X_j.permute(0, 3, 1, 2),  # (B, 3, H, W)
            corresponding_coords_norm,
            mode='bilinear', padding_mode='border', align_corners=True
        ).permute(0, 2, 3, 1)  # (B, H, W, 3)
        
        # 计算场景光流: S_{i→j} = X_{j,i} - X_{i,i}
        # 这表示3D点在世界坐标系中的真实移动
        scene_flow = X_j_sampled - X_i  # (B, H, W, 3)
        
        return scene_flow
    
    def _compute_theoretical_flow(self, X_i, scene_flow, pose_i, pose_j, K):
        """
        计算理论光流 f̂_{i→j} (Endo3R公式5)
        f̂_{i→j}(u') = KT_{j,i}(X_{i,i}(u') + S_{i→j}(u')) - u'
        
        物理含义：基于估计的深度、位姿和场景运动，理论上应该产生的光流
        
        Args:
            X_i: 帧i的3D点云 (B, H, W, 3)
            scene_flow: 场景光流 (B, H, W, 3)
            pose_i: 帧i的相机位姿 (B, 4, 4) camera-to-world
            pose_j: 帧j的相机位姿 (B, 4, 4) camera-to-world
            K: 相机内参矩阵 (B, 3, 3)
            
        Returns:
            theoretical_flow: 理论光流 (B, 2, H, W)
        """
        B, H, W, _ = X_i.shape
        device = X_i.device
        
        # pose_i和pose_j是camera-to-world变换矩阵
        # 但我们需要将3D点从相机i坐标系变换到相机j坐标系
        # T_{cam_j <- cam_i} = T_{cam_j <- world} @ T_{world <- cam_i} = inv(pose_j) @ pose_i
        from dust3r.utils.geometry import inv
        relative_pose = torch.bmm(inv(pose_j), pose_i)  # (B, 4, 4)
        
        # 加上场景运动得到点的新位置：X_{i,i} + S_{i→j}
        X_i_with_motion = X_i + scene_flow  # (B, H, W, 3)
        
        # 转换为齐次坐标并应用相对位姿变换
        X_i_flat = X_i_with_motion.view(B, -1, 3)  # (B, H*W, 3)
        ones = torch.ones(B, H*W, 1, device=device)
        X_i_homo = torch.cat([X_i_flat, ones], dim=-1)  # (B, H*W, 4)
        
        # 应用相对位姿变换: T_{j,i} @ X_i_homo^T
        X_j_homo = torch.bmm(relative_pose, X_i_homo.transpose(1, 2))  # (B, 4, H*W)
        X_j_transformed = X_j_homo[:, :3, :].transpose(1, 2)  # (B, H*W, 3)
        
        # 相机投影: K @ X_j_transformed
        projected = torch.bmm(K, X_j_transformed.transpose(1, 2))  # (B, 3, H*W)
        projected = projected.transpose(1, 2).view(B, H, W, 3)  # (B, H, W, 3)
        
        # 透视除法得到像素坐标
        z = projected[..., 2:3].clamp(min=1e-6)
        pixel_coords = projected[..., :2] / z  # (B, H, W, 2)
        
        # 构建原始像素网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        original_coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        
        # 计算理论光流: f̂ = pixel_coords - original_coords
        theoretical_flow = (pixel_coords - original_coords).permute(0, 3, 1, 2)  # (B, 2, H, W)
        
        return theoretical_flow
    
    def compute_loss(self, gts, preds, **kw):
        """
        计算Endo3R动态感知光流损失
        
        Args:
            gts: 真实数据列表，包含图像
            preds: 预测数据列表，包含3D点云和位姿
            
        Returns:
            (loss, details): 损失值和详细信息
        """
        if len(gts) < 2 or self.raft_model is None:
            device = preds[0]["pts3d_in_self_view"].device
            return torch.tensor(0.0, device=device), {"endo3r_disabled": True}
        
        total_loss = 0.0
        num_pairs = 0
        details = {}
        
        # 根据策略选择视角对
        if self.flow_strategy == "anchor_based":
            # 策略A: 所有视角与第一视角计算 (与ConfLoss一致)
            # 视角对: (1->0), (2->0), ..., (N-1->0)
            view_pairs = [(i, 0) for i in range(1, len(gts))]
            details["flow_strategy"] = "anchor_based"
            
        elif self.flow_strategy == "consecutive":
            # 策略B: 连续帧对 (原实现)
            # 视角对: (0->1), (1->2), ..., (N-2->N-1)
            view_pairs = [(i, i+1) for i in range(len(gts) - 1)]
            details["flow_strategy"] = "consecutive"
            
        elif self.flow_strategy == "mixed":
            # 策略C: 混合策略
            # 主要约束: 所有视角->第一视角，权重1.0
            # 辅助约束: 相邻帧对，权重0.5
            view_pairs = []
            # 主要约束
            for i in range(1, len(gts)):
                view_pairs.append((i, 0, 1.0))  # (source, target, weight)
            # 辅助约束  
            for i in range(len(gts) - 1):
                view_pairs.append((i, i+1, 0.5))
            details["flow_strategy"] = "mixed"
            
        else:
            raise ValueError(f"Unknown flow_strategy: {self.flow_strategy}")
        
        # 处理视角对
        for pair_info in view_pairs:
            if len(pair_info) == 2:
                i, j = pair_info
                weight = 1.0
            else:
                i, j, weight = pair_info
            
            try:
                # === 数据提取 ===
                img_i = gts[i]["img"]  # (B, H, W, 3)
                img_j = gts[j]["img"]  # (B, H, W, 3)
                X_i = preds[i]["pts3d_in_self_view"]  # (B, H, W, 3)
                X_j = preds[j]["pts3d_in_self_view"]  # (B, H, W, 3)
                
                # 获取位姿（camera-to-world）
                from dust3r.utils.camera import pose_encoding_to_camera
                pose_i = pose_encoding_to_camera(preds[i]["camera_pose"])  # (B, 4, 4)
                pose_j = pose_encoding_to_camera(preds[j]["camera_pose"])  # (B, 4, 4)
                
                # === 步骤1: 计算观察光流 O_{i→j} ===
                observed_flow = self._compute_observed_flow(img_i, img_j)  # (B, 2, H, W)
                if observed_flow is None:
                    continue
                
                # === 步骤2: 估计内参矩阵 ===
                K = self._estimate_intrinsics_from_pointcloud(X_i)  # (B, 3, 3)
                
                # === 步骤3: 计算场景光流 S_{i→j} (Endo3R公式4) ===
                scene_flow = self._compute_scene_flow(X_i, X_j, observed_flow)  # (B, H, W, 3)
                
                # === 步骤4: 计算理论光流 f̂_{i→j} (Endo3R公式5) ===
                theoretical_flow = self._compute_theoretical_flow(
                    X_i, scene_flow, pose_i, pose_j, K
                )  # (B, 2, H, W)
                
                # === 步骤5: 计算光流一致性损失 (Endo3R公式6) ===
                flow_diff = theoretical_flow - observed_flow
                pair_loss = torch.abs(flow_diff).mean()  # L1损失
                
                # 应用权重
                weighted_pair_loss = pair_loss * weight
                total_loss += weighted_pair_loss
                num_pairs += 1
                
                # 记录详细信息 - 确保转换为float
                details[f"endo3r_flow_loss_{i}_{j}"] = float(pair_loss.detach().cpu().item()) if hasattr(pair_loss, 'item') else float(pair_loss)
                if weight != 1.0:
                    details[f"endo3r_flow_loss_{i}_{j}_weighted"] = float(weighted_pair_loss.detach().cpu().item()) if hasattr(weighted_pair_loss, 'item') else float(weighted_pair_loss)
                
                # 可视化信息（仅保存第一对的统计信息，避免tensor导致的AssertionError）
                if i == 0:
                    details["endo3r_observed_flow_mean"] = float(observed_flow.abs().mean().detach().cpu().item())
                    details["endo3r_theoretical_flow_mean"] = float(theoretical_flow.abs().mean().detach().cpu().item())
                    details["endo3r_scene_flow_mean"] = float(scene_flow.abs().mean().detach().cpu().item())
                
            except Exception as e:
                print(f"⚠️ Endo3R损失计算失败 (帧{i}-{j}): {e}")
                continue
        
        # 计算平均损失
        if num_pairs > 0:
            avg_loss = total_loss / num_pairs
        else:
            device = preds[0]["pts3d_in_self_view"].device
            avg_loss = torch.tensor(0.0, device=device)
        
        # 添加汇总信息 - 确保转换为float
        details["endo3r_avg_loss"] = float(avg_loss.detach().cpu().item()) if hasattr(avg_loss, 'item') else float(avg_loss)
        details["endo3r_valid_pairs"] = int(num_pairs)
        
        # 计算总帧对数（根据策略而定）
        if self.flow_strategy == "anchor_based":
            total_pairs = len(gts) - 1  # (1->0), (2->0), ..., (N-1->0)
        elif self.flow_strategy == "consecutive":
            total_pairs = len(gts) - 1  # (0->1), (1->2), ..., (N-2->N-1)
        elif self.flow_strategy == "mixed":
            total_pairs = (len(gts) - 1) + (len(gts) - 1)  # anchor + consecutive
        else:
            total_pairs = len(view_pairs)
            
        details["endo3r_total_pairs"] = int(total_pairs)
        
        return avg_loss, details