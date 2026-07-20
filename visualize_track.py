"""
Visualize 3D Tracking on Rendered Reconstruction Video

Takes the 3D trajectory from track_3d.py and overlays it on the rendered
reconstruction video as a colored marker, showing the FLIM-marked tissue
being tracked directly on the 3D reconstruction output.

Usage:
    python visualize_track.py \
        --model_path output/endonerf/pulling \
        --configs arguments/endonerf/pulling.py \
        --trajectory tracked_trajectory.npy \
        --output tracked_video.mp4

This script:
1. Loads the trained model and renders each frame
2. Projects the 3D tracked point onto each rendered frame
3. Draws a colored circle at the projected location
4. Saves the annotated video
"""

import torch
import numpy as np
import math
import os
import cv2
import sys
from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal


def project_to_2d(point_3d, camera):
    """Project a 3D world point to pixel coordinates."""
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0

    w2c = camera.world_view_transform.transpose(0, 1)
    pt_world = torch.tensor([point_3d[0], point_3d[1], point_3d[2], 1.0],
                            dtype=torch.float32, device=w2c.device)
    pt_cam = (w2c @ pt_world)[:3]

    if pt_cam[2] <= 0:
        return None, None

    u = fx * pt_cam[0] / pt_cam[2] + cx
    v = fy * pt_cam[1] / pt_cam[2] + cy
    return int(round(u.item())), int(round(v.item()))


def main():
    parser = ArgumentParser(description="Visualize 3D tracking on reconstruction video")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--quiet", action="store_true")

    # Visualization arguments
    parser.add_argument("--trajectory", type=str, required=True,
                        help="Path to 3D trajectory .npy from track_3d.py")
    parser.add_argument("--output", type=str, default="tracked_reconstruction.mp4",
                        help="Output video path")
    parser.add_argument("--marker_radius", type=int, default=8,
                        help="Radius of the tracking marker in pixels")
    parser.add_argument("--marker_color", type=int, nargs=3, default=[0, 255, 0],
                        help="Marker color in BGR (default: green)")
    parser.add_argument("--show_trail", action="store_true",
                        help="Draw trailing path of previous positions")
    parser.add_argument("--trail_length", type=int, default=20,
                        help="Number of past frames to show in trail")
    parser.add_argument("--use_train", action="store_true",
                        help="Use training cameras instead of video cameras")
    parser.add_argument("--fps", type=int, default=30)

    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)

    # Load trajectory
    trajectory_3d = np.load(args.trajectory)
    print(f"Loaded 3D trajectory: {trajectory_3d.shape}")

    # Load model
    print(f"Loading model from {args.model_path}")
    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree, hyperparam.extract(args))
        scene = Scene(model.extract(args), gaussians,
                      load_iteration=args.iteration, shuffle=False)

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.use_train:
            cameras = scene.getTrainCameras()
        else:
            cameras = scene.getVideoCameras()
        print(f"Rendering {len(cameras)} frames")

        # Build timestamp-to-trajectory-index mapping
        train_cameras = scene.getTrainCameras()
        timestamps = sorted(set(cam.time for cam in train_cameras))
        time_to_idx = {t: i for i, t in enumerate(timestamps)}

        # Setup video writer
        first_cam = cameras[0]
        h, w = first_cam.image_height, first_cam.image_width
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

        marker_color = tuple(args.marker_color)
        trail_points = []

        for idx, cam in enumerate(tqdm(cameras, desc="Rendering + tracking")):
            # Render the frame
            render_pkg = render(cam, gaussians, pipeline.extract(args),
                               background, stage="fine")
            image = render_pkg["render"]  # [3, H, W]

            # Convert to numpy BGR for OpenCV
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Find the trajectory point for this frame's time
            t = cam.time
            # Find nearest timestamp in our trajectory
            t_idx = min(range(len(timestamps)),
                        key=lambda i: abs(timestamps[i] - t))
            if t_idx < len(trajectory_3d):
                pt_3d = trajectory_3d[t_idx]

                # Project to this camera's 2D view
                u_px, v_px = project_to_2d(pt_3d, cam)

                if u_px is not None and 0 <= u_px < w and 0 <= v_px < h:
                    # Draw trailing path
                    if args.show_trail:
                        trail_points.append((u_px, v_px))
                        if len(trail_points) > args.trail_length:
                            trail_points = trail_points[-args.trail_length:]
                        for i, (tx, ty) in enumerate(trail_points[:-1]):
                            alpha = (i + 1) / len(trail_points)
                            trail_color = tuple(int(c * alpha * 0.6) for c in marker_color)
                            cv2.circle(img_bgr, (tx, ty), 3, trail_color, -1)

                    # Draw main marker
                    cv2.circle(img_bgr, (u_px, v_px), args.marker_radius,
                              marker_color, -1)
                    cv2.circle(img_bgr, (u_px, v_px), args.marker_radius + 2,
                              (255, 255, 255), 1)

                    # Label with frame number
                    cv2.putText(img_bgr, f"F{idx}", (u_px + 12, v_px - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            writer.write(img_bgr)

        writer.release()
        print(f"\nSaved tracked reconstruction video to: {args.output}")


if __name__ == "__main__":
    main()
