"""
Overlay ALL FLIm markers (every tracked query, across the whole video) onto a single
frame's point cloud -- the complete picture of every tissue spot the FLIm laser touched,
rather than only whichever subset happens to be seeded-and-visible at one moment (which is
what overlay_tracks.py's per-frame accumulation shows).

Rewritten to match the CURRENT per-query pipeline (extract_tracks.py / overlay_tracks.py):
tracks/visibility are [T, N, 2] / [T, N] (N = number of distinct FLIm detections, each its
own CoTracker query), not the old collapsed single-path [T, 2] / [T] this script used to
expect.

Each query's marker is placed at its CoTracker-predicted pixel position AT THE TARGET FRAME
(tracks[target_frame, i] -- CoTracker predicts this for every query at every frame regardless
of when it was seeded, confirmed empirically), backprojected using the TARGET frame's OWN
depth map. This guarantees, by construction, that every marker lands exactly on the target
frame's actual reconstructed surface, since it's built from the exact same depth values the
base point cloud itself was built from.

An earlier version of this instead backprojected at the query's OWN seed frame and
reprojected that camera-space point into the target frame's camera space via
world_view_transforms (i.e. correcting only for camera motion between the two frames). That
did not fix markers rendering behind the reconstruction, because the mismatch isn't caused by
camera motion -- it's that the tissue itself deforms between a query's seed frame and the
target frame, so a depth value recorded at frame S has no reason to still describe the
surface at frame 40. Reading the depth directly off the target frame at the query's
target-frame-tracked pixel sidesteps that entirely: right or wrong about where the marker
"really" is over time, it is always consistent with what's actually on screen at the target
frame. The seed frame is still used for the CSV color lookup (which WLI_4 reading applies to
this query), just not for its 3D position.

Usage:
    python overlay_all_traces.py \
        --reconstruct_dir reconstruct/headAndNeck/P202/run7_raft_stereo_pruned_v5_more_fixes_video \
        --tracks data/headAndNeck/tracks/P202/run7/cotracker_tracks.npy \
        --visibility data/headAndNeck/tracks/P202/run7/cotracker_visibility.npy \
        --seed_frames data/headAndNeck/tracks/P202/run7/cotracker_seed_frames.npy \
        --csv_subject_id 120 --csv_run 3 \
        --target_frame 40 \
        --output frame_40_all_traces.ply
"""

import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

from overlay_tracks import backproject_pixel, accumulate_marker_blend, finalize_blend, flim_marker_color


def main():
    parser = ArgumentParser(description="Overlay ALL FLIm queries onto a single frame's point cloud")
    parser.add_argument("--reconstruct_dir", type=str, required=True,
                        help="Output of `render.py --reconstruct` -- must contain frame_<i>.ply "
                             "plus depths.npy/camera_params.npy/frame_indices.npy/frame_uids.npy")
    parser.add_argument("--tracks", type=str, required=True,
                        help="Per-query CoTracker tracks .npy, shape [T, N, 2], from extract_tracks.py")
    parser.add_argument("--visibility", type=str, default="",
                        help="Per-query visibility .npy [T, N] (only used to sanity-check the "
                             "seed frame itself; unused if omitted)")
    parser.add_argument("--seed_frames", type=str, required=True,
                        help="Per-query seed frame .npy [N] from extract_tracks.py -- each query's "
                             "marker is placed using ITS OWN seed frame's depth/pose, the one frame "
                             "its detection is guaranteed valid at")
    parser.add_argument("--target_frame", type=int, default=40,
                        help="ply filename (frame_<target_frame>.ply) whose point cloud all markers "
                             "get stamped onto")
    parser.add_argument("--output", type=str, default="frame_all_traces.ply")
    parser.add_argument("--marker_radius", type=float, default=None,
                        help="Reference radius for a marker's overall size -- see "
                             "--solid_core_fraction/--blend_radius_factor for how it's actually "
                             "split into a solid core vs. a blend halo. Default: auto-scaled to "
                             "0.22%% of the target frame's own bounding-box diagonal (see "
                             "overlay_tracks.py)")
    parser.add_argument("--solid_core_fraction", type=float, default=0.1,
                        help="Real cloud points within marker_radius*this fraction are fully "
                             "solid marker color; beyond that (out to the blend halo) they're "
                             "linearly recolored. Lowered from an implicit 1.0 (the entire core "
                             "was solid, blending only started AT marker_radius) -- that made "
                             "the solid region dominate the marker's visible area, so the blend "
                             "was too thin a sliver at the very edge to notice. At 0.1, blending "
                             "starts almost immediately outside the center instead.")
    parser.add_argument("--blend_radius_factor", type=float, default=2.5,
                        help="Real cloud points beyond the solid core but within "
                             "marker_radius*this factor are linearly recolored from the marker "
                             "color (at the core's edge) to their OWN original color (at the "
                             "halo's outer edge) -- softens the marker into its real surroundings "
                             "instead of a hard color boundary. Set to 1.0 to disable (no halo, "
                             "just the solid core).")
    parser.add_argument("--blend_ease_power", type=float, default=0.4,
                        help="Exponent applied to the halo's distance-based blend fraction "
                             "before mixing colors (<1 front-loads the transition) -- see "
                             "overlay_tracks.blend_recolor_nearby")
    parser.add_argument("--flim_csv", type=str,
                        default="/media/data/zliwang/data/HN_P2C_20250304.csv")
    parser.add_argument("--csv_subject_id", type=int, required=True)
    parser.add_argument("--csv_run", type=int, required=True)
    parser.add_argument("--csv_frame_stride", type=int, default=1,
                        help="seed_frames[i]*csv_frame_stride + csv_frame_offset gives the CSV's "
                             "WLI_1 frame index -- see overlay_tracks.py")
    parser.add_argument("--csv_frame_offset", type=int, default=0)
    args = parser.parse_args()

    depths = np.load(os.path.join(args.reconstruct_dir, "depths.npy"))
    frame_idxs = np.load(os.path.join(args.reconstruct_dir, "frame_indices.npy"))
    focal_x, focal_y, width, height = np.load(os.path.join(args.reconstruct_dir, "camera_params.npy"))
    width, height = int(width), int(height)
    k_target = int(np.where(frame_idxs == args.target_frame)[0][0])
    target_depth = depths[k_target]

    tracks = np.load(args.tracks)
    assert tracks.ndim == 3 and tracks.shape[2] == 2, f"Expected [T, N, 2], got {tracks.shape}"
    visibility = np.load(args.visibility) if args.visibility else None
    seed_frames = np.load(args.seed_frames)
    T, N, _ = tracks.shape
    assert seed_frames.shape == (N,), f"Expected seed_frames shape ({N},), got {seed_frames.shape}"

    flim_df = pd.read_csv(args.flim_csv, low_memory=False)
    flim_df = flim_df[(flim_df["SubjectID"] == args.csv_subject_id) &
                       (flim_df["Run"] == args.csv_run)]
    print(f"Loaded {len(flim_df)} CSV rows for SubjectID={args.csv_subject_id} Run={args.csv_run}")

    target_ply = os.path.join(args.reconstruct_dir, f"frame_{args.target_frame}.ply")
    base_pcd = o3d.io.read_point_cloud(target_ply)
    base_points = np.asarray(base_pcd.points)
    base_colors = np.asarray(base_pcd.colors)
    print(f"Base frame {args.target_frame}: {len(base_points)} points")

    if args.marker_radius is not None:
        radius = args.marker_radius
    elif len(base_points) > 0:
        radius = (0.02 / 9.0) * np.linalg.norm(base_points.max(axis=0) - base_points.min(axis=0))
    else:
        radius = 0.005

    solid_radius = radius * args.solid_core_fraction
    blend_radius = radius * args.blend_radius_factor
    tree = cKDTree(base_points) if len(base_points) > 0 else None
    original_colors = base_colors.copy()
    weighted_color_sum = np.zeros_like(base_colors)
    weight_sum = np.zeros(len(base_colors))

    n_placed = 0
    skip_no_depth = skip_no_label = 0

    for i in range(N):
        seed_frame = int(seed_frames[i])

        full_frame = seed_frame * args.csv_frame_stride + args.csv_frame_offset
        color = flim_marker_color(flim_df, full_frame)
        if color is None:
            skip_no_label += 1
            continue

        uv = tracks[args.target_frame, i]
        cam_pt = backproject_pixel(target_depth, uv[0], uv[1], focal_x, focal_y, width, height)
        if cam_pt is None:
            skip_no_depth += 1
            continue

        if tree is not None:
            accumulate_marker_blend(
                tree, weighted_color_sum, weight_sum, cam_pt, solid_radius, blend_radius, color,
                ease_power=args.blend_ease_power)
        n_placed += 1

    n_touched_total = finalize_blend(base_colors, original_colors, weighted_color_sum, weight_sum) \
        if tree is not None else 0

    print(f"Placed {n_placed}/{N} FLIm markers onto frame {args.target_frame} "
          f"({skip_no_label} skipped: no matching CSV label, "
          f"{skip_no_depth} skipped: no valid depth at the target frame's tracked pixel), "
          f"{n_touched_total} real points recolored")

    if n_placed == 0:
        print("No markers placed.")
        return

    base_pcd.colors = o3d.utility.Vector3dVector(base_colors)
    o3d.io.write_point_cloud(args.output, base_pcd)
    print(f"Saved to: {args.output}")
    print(f"Total points: {len(base_points)} (unchanged -- markers recolor real points in place, "
          f"no synthetic points added)")


if __name__ == "__main__":
    main()
