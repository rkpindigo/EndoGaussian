"""
3D Tissue Tracking via EndoGaussian Deformation Network

Given a pixel coordinate (u, v) in a reference frame where the FLIM-marked tissue
is visible, this script:
1. Loads the trained EndoGaussian model
2. Renders the reference frame to get its depth map
3. Unprojects the pixel to 3D world coordinates
4. Finds the K nearest Gaussians to that 3D point
5. Queries the deformation network at every timestep to get the 3D trajectory
6. Saves the trajectory and optionally visualizes it

Usage:
    python track_3d.py \
        --model_path output/endonerf/pulling \
        --configs arguments/endonerf/pulling.py \
        --pixel 320 240 \
        --ref_frame 0 \
        --output tracked_trajectory.npy
"""

import torch
import numpy as np
import math
import os
import sys
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal


def unproject_pixel(u, v, depth, camera):
    """Unproject a pixel (u, v) with known depth to 3D world coordinates."""
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0

    # Camera-space 3D point
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    # Camera-to-world transform
    # world_view_transform is stored as W2C transposed (column-major convention)
    w2c = camera.world_view_transform.transpose(0, 1)  # [4, 4]
    c2w = torch.inverse(w2c)

    pt_cam = torch.tensor([x_cam, y_cam, z_cam, 1.0], dtype=torch.float32)
    pt_world = (c2w @ pt_cam)[:3]

    return pt_world


def find_nearest_gaussians(target_3d, gaussians, k=5):
    """Find the K nearest Gaussians to a 3D world point."""
    means3D = gaussians.get_xyz  # [N, 3]
    target = target_3d.to(means3D.device)
    dists = torch.norm(means3D - target.unsqueeze(0), dim=1)
    topk = torch.topk(dists, k, largest=False)
    return topk.indices, topk.values


def query_deformation_at_time(gaussians, gaussian_indices, t):
    """Query the deformation network for specific Gaussians at time t.

    Returns the deformed 3D positions of the selected Gaussians.
    """
    means3D = gaussians._xyz[gaussian_indices]
    scales = gaussians._scaling[gaussian_indices]
    rotations = gaussians._rotation[gaussian_indices]
    opacity = gaussians._opacity[gaussian_indices]
    time_input = torch.ones(means3D.shape[0], 1, device=means3D.device) * t

    deformed_means, _, _, _ = gaussians._deformation(
        means3D, scales, rotations, opacity, times_sel=time_input
    )
    return deformed_means


def compute_trajectory(gaussians, gaussian_indices, weights, timestamps):
    """Compute weighted 3D trajectory across all timestamps."""
    trajectory = []

    for t in timestamps:
        deformed = query_deformation_at_time(gaussians, gaussian_indices, t)
        # Weighted average of K nearest Gaussians
        weighted_pos = (deformed * weights.unsqueeze(1)).sum(dim=0)
        trajectory.append(weighted_pos.detach().cpu().numpy())

    return np.array(trajectory)  # [T, 3]


def project_to_2d(point_3d, camera):
    """Project a 3D world point back to pixel coordinates for a given camera."""
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0

    w2c = camera.world_view_transform.transpose(0, 1)  # [4, 4]
    pt_world = torch.tensor([*point_3d, 1.0], dtype=torch.float32, device=w2c.device)
    pt_cam = (w2c @ pt_world)[:3]

    u = fx * pt_cam[0] / pt_cam[2] + cx
    v = fy * pt_cam[1] / pt_cam[2] + cy
    return u.item(), v.item()


def main():
    parser = ArgumentParser(description="3D tissue tracking via deformation network")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--quiet", action="store_true")

    # Tracking-specific arguments
    parser.add_argument("--pixel", type=int, nargs=2, required=True,
                        help="Pixel coordinate (u, v) of the FLIM mark in the reference frame")
    parser.add_argument("--ref_frame", type=int, default=0,
                        help="Reference frame index where the mark is visible")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of nearest Gaussians to associate with the mark")
    parser.add_argument("--output", type=str, default="tracked_trajectory.npy",
                        help="Output path for the 3D trajectory array [T, 3]")
    parser.add_argument("--output_2d", type=str, default="",
                        help="If set, also save reprojected 2D pixel trajectory [T, 2]")
    parser.add_argument("--cotracker_tracks", type=str, default="",
                        help="Path to CoTracker tracks .npy [T, 2] for comparison")

    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)

    u, v = args.pixel
    ref_idx = args.ref_frame
    top_k = args.top_k

    # --- Load trained model ---
    print(f"Loading model from {args.model_path}")
    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree, hyperparam.extract(args))
        scene = Scene(model.extract(args), gaussians,
                      load_iteration=args.iteration, shuffle=False)

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_cameras = scene.getTrainCameras()
        print(f"Loaded {len(train_cameras)} training cameras, "
              f"{gaussians.get_xyz.shape[0]} Gaussians")

        # --- Render reference frame to get depth ---
        ref_camera = train_cameras[ref_idx]
        print(f"Reference frame {ref_idx}: time={ref_camera.time:.4f}, "
              f"image={ref_camera.image_width}x{ref_camera.image_height}")

        render_pkg = render(ref_camera, gaussians, pipeline.extract(args),
                           background, stage="fine")
        depth_map = render_pkg["depth"]  # [1, H, W]

        # Get depth at the target pixel
        depth_val = depth_map[0, v, u].item()
        if depth_val <= 0:
            print(f"WARNING: depth at pixel ({u}, {v}) is {depth_val}. "
                  f"Trying nearby pixels...")
            # Search in a small window for valid depth
            for radius in range(1, 10):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = v + dy, u + dx
                        if 0 <= ny < depth_map.shape[1] and 0 <= nx < depth_map.shape[2]:
                            d = depth_map[0, ny, nx].item()
                            if d > 0:
                                depth_val = d
                                u, v = nx, ny
                                break
                    if depth_val > 0:
                        break
                if depth_val > 0:
                    break
            if depth_val <= 0:
                print("ERROR: Could not find valid depth near the target pixel.")
                sys.exit(1)

        print(f"Depth at pixel ({u}, {v}): {depth_val:.4f}")

        # --- Unproject to 3D ---
        target_3d = unproject_pixel(u, v, depth_val, ref_camera)
        print(f"Unprojected 3D point: [{target_3d[0]:.4f}, {target_3d[1]:.4f}, {target_3d[2]:.4f}]")

        # --- Find nearest Gaussians ---
        indices, distances = find_nearest_gaussians(target_3d, gaussians, k=top_k)
        print(f"Found {top_k} nearest Gaussians (distances: "
              f"{distances.cpu().numpy().round(4)})")

        # Compute inverse-distance weights (closer Gaussians contribute more)
        inv_dists = 1.0 / (distances + 1e-8)
        weights = inv_dists / inv_dists.sum()
        print(f"Weights: {weights.cpu().numpy().round(4)}")

        # --- Collect all timestamps from training cameras ---
        timestamps = sorted(set(cam.time for cam in train_cameras))
        print(f"Tracking across {len(timestamps)} unique timestamps "
              f"[{min(timestamps):.4f} -> {max(timestamps):.4f}]")

        # --- Query deformation network at all timestamps ---
        trajectory_3d = compute_trajectory(gaussians, indices, weights, timestamps)
        print(f"3D trajectory shape: {trajectory_3d.shape}")

        # --- Save 3D trajectory ---
        np.save(args.output, trajectory_3d)
        print(f"Saved 3D trajectory to: {args.output}")

        # --- Optionally compute and save 2D reprojection ---
        if args.output_2d:
            trajectory_2d = []
            for i, cam in enumerate(train_cameras):
                t_idx = timestamps.index(cam.time)
                pt_3d = trajectory_3d[t_idx]
                u_proj, v_proj = project_to_2d(pt_3d, cam)
                trajectory_2d.append([u_proj, v_proj])
            trajectory_2d = np.array(trajectory_2d)
            np.save(args.output_2d, trajectory_2d)
            print(f"Saved 2D reprojected trajectory to: {args.output_2d}")

        # --- Optionally compare with CoTracker ---
        if args.cotracker_tracks:
            cotracker = np.load(args.cotracker_tracks)
            print(f"\nCoTracker tracks shape: {cotracker.shape}")
            if args.output_2d:
                diff = trajectory_2d[:len(cotracker)] - cotracker[:len(trajectory_2d)]
                rmse = np.sqrt((diff ** 2).sum(axis=1).mean())
                print(f"RMSE vs CoTracker (pixels): {rmse:.2f}")

        # --- Print summary ---
        displacement = np.linalg.norm(trajectory_3d - trajectory_3d[0], axis=1)
        print(f"\n--- Tracking Summary ---")
        print(f"Total 3D displacement range: {displacement.min():.4f} - {displacement.max():.4f}")
        print(f"Max distance from start: {displacement.max():.4f}")
        print(f"Final position: [{trajectory_3d[-1][0]:.4f}, "
              f"{trajectory_3d[-1][1]:.4f}, {trajectory_3d[-1][2]:.4f}]")


if __name__ == "__main__":
    main()
