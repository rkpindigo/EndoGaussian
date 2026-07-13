import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

class HexPlaneField(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ])
        self.aabb = nn.Parameter(aabb,requires_grad=True) # !!!!!
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
            
        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features

class WindowedHexPlaneField(nn.Module):
    """HexPlane with temporal windowing for long sequences.

    Spatial planes (XY, XZ, YZ) are shared globally.
    Temporal planes (XT, YT, ZT) are per-window with overlap blending.
    """
    def __init__(self, bounds, planeconfig, multires, window_size=100, window_stride=50, n_frames=608):
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)

        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.window_size = window_size
        self.window_stride = window_stride

        n_windows = max(1, 1 + (n_frames - window_size) // window_stride)
        self.n_windows = n_windows
        self.n_frames = n_frames
        print(f"WindowedHexPlane: {n_windows} windows, size={window_size}, stride={window_stride}, "
              f"n_frames={n_frames}")

        # 6 planes per scale: indices 0=XY, 1=XZ, 2=XT, 3=YZ, 4=YT, 5=ZT
        # Spatial planes: 0 (XY), 1 (XZ), 3 (YZ) — shared
        # Temporal planes: 2 (XT), 4 (YT), 5 (ZT) — per window
        self.spatial_plane_indices = [0, 1, 3]
        self.temporal_plane_indices = [2, 4, 5]

        self.spatial_grids = nn.ModuleList()
        self.window_grids = nn.ModuleList()
        self.feat_dim = 0

        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            spatial_reso = [r * res for r in config["resolution"][:3]]
            out_dim = config["output_coordinate_dim"]

            # Shared spatial planes (XY, XZ, YZ)
            spatial_params = nn.ParameterList()
            coo_combs = list(itertools.combinations(range(4), 2))
            for ci in self.spatial_plane_indices:
                coo_comb = coo_combs[ci]
                plane_reso = [spatial_reso[cc] for cc in coo_comb[::-1]]
                p = nn.Parameter(torch.empty([1, out_dim] + plane_reso))
                nn.init.uniform_(p, a=0.1, b=0.5)
                spatial_params.append(p)
            self.spatial_grids.append(spatial_params)

            # Per-window temporal planes (XT, YT, ZT)
            per_scale_windows = nn.ModuleList()
            for _ in range(n_windows):
                window_params = nn.ParameterList()
                for ci in self.temporal_plane_indices:
                    coo_comb = coo_combs[ci]
                    plane_reso = []
                    for cc in coo_comb[::-1]:
                        if cc == 3:
                            plane_reso.append(window_size)
                        else:
                            plane_reso.append(spatial_reso[cc])
                    p = nn.Parameter(torch.empty([1, out_dim] + plane_reso))
                    nn.init.ones_(p)
                    window_params.append(p)
                per_scale_windows.append(window_params)
            self.window_grids.append(per_scale_windows)

            if self.concat_features:
                self.feat_dim += out_dim
            else:
                self.feat_dim = out_dim

        print("feature_dim:", self.feat_dim)

    def _get_window_info(self, t_global):
        """For a global time in [0,1], find which window(s) it falls in and local coords."""
        t_frame = t_global * (self.n_frames - 1)

        results = []
        for w in range(self.n_windows):
            w_start = w * self.window_stride
            w_end = w_start + self.window_size - 1
            if t_frame < w_start - 0.5 or t_frame > w_end + 0.5:
                continue
            t_local = (t_frame - w_start) / max(self.window_size - 1, 1)
            t_local = t_local * 2.0 - 1.0  # map to [-1, 1] for grid_sample
            t_local = max(-1.0, min(1.0, t_local))
            results.append((w, t_local))

        if len(results) == 0:
            last_w = self.n_windows - 1
            results = [(last_w, 1.0)]

        return results

    def _query_single_window(self, pts_norm, t_local, window_idx, scale_id):
        """Query one window: multiply spatial and temporal plane features."""
        coo_combs = list(itertools.combinations(range(4), 2))
        pts_4d = torch.cat([pts_norm, t_local * torch.ones_like(pts_norm[:, :1])], dim=-1)

        interp_space = 1.0
        # Spatial planes
        for si, ci in enumerate(self.spatial_plane_indices):
            coo_comb = coo_combs[ci]
            feature_dim = self.spatial_grids[scale_id][si].shape[1]
            interp_out = grid_sample_wrapper(
                self.spatial_grids[scale_id][si], pts_4d[..., coo_comb]
            ).view(-1, feature_dim)
            interp_space = interp_space * interp_out

        # Temporal planes from this window
        for ti, ci in enumerate(self.temporal_plane_indices):
            coo_comb = coo_combs[ci]
            feature_dim = self.window_grids[scale_id][window_idx][ti].shape[1]
            interp_out = grid_sample_wrapper(
                self.window_grids[scale_id][window_idx][ti], pts_4d[..., coo_comb]
            ).view(-1, feature_dim)
            interp_space = interp_space * interp_out

        return interp_space

    def forward(self, pts, timestamps=None):
        if pts.shape[0] == 0:
            return torch.zeros((0, self.feat_dim), device=pts.device)

        pts_norm = normalize_aabb(pts, self.aabb)

        # All points share the same timestamp (single frame render)
        t_val = timestamps[0, 0].item() if timestamps.dim() > 1 else timestamps[0].item()
        window_info = self._get_window_info(t_val)

        multi_scale_interp = []
        for scale_id in range(len(self.multiscale_res_multipliers)):
            if len(window_info) == 1:
                w_idx, t_local = window_info[0]
                scale_feat = self._query_single_window(pts_norm, t_local, w_idx, scale_id)
            else:
                # Blend between overlapping windows
                w0, t0 = window_info[0]
                w1, t1 = window_info[1]

                # Blend weight: how far into the overlap region
                overlap_start = w1 * self.window_stride
                overlap_end = w0 * self.window_stride + self.window_size - 1
                t_frame = t_val * (self.n_frames - 1)
                if overlap_end > overlap_start:
                    alpha = (t_frame - overlap_start) / (overlap_end - overlap_start)
                else:
                    alpha = 0.5
                alpha = max(0.0, min(1.0, alpha))

                feat0 = self._query_single_window(pts_norm, t0, w0, scale_id)
                feat1 = self._query_single_window(pts_norm, t1, w1, scale_id)
                scale_feat = (1.0 - alpha) * feat0 + alpha * feat1

            multi_scale_interp.append(scale_feat)

        return torch.cat(multi_scale_interp, dim=-1) if self.concat_features else sum(multi_scale_interp)

    def get_all_grid_params(self):
        """Returns all grid parameters (spatial + all windows) as a flat list for regulation."""
        params = []
        for sp in self.spatial_grids:
            params.extend(list(sp))
        for per_scale in self.window_grids:
            for window in per_scale:
                params.extend(list(window))
        return params

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min])
        self.aabb = nn.Parameter(aabb, requires_grad=True)
        print("Voxel Plane: set aabb=", self.aabb)