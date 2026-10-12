"""
Variant of overlay_all_traces.py that does NOT look up real FLIm severity from the CSV.
Instead, each tracked query is randomly assigned one of the 6 FLIM_COLOR_SCALE severity-bucket
colors (overlay_tracks.FLIM_COLOR_SCALE -- the Green To Blue Gradient scheme, green=healthy
through blue=cancer), so every marker still ends up at one of the same discrete severity levels
overlay_tracks.py/overlay_all_traces.py use, but the bucket is randomly assigned rather than
matched to an actual WLI_4 reading. Useful for visually spot-checking marker placement/density
across the whole reconstruction without needing (or being limited by) real CSV label coverage
-- every query gets a marker, not just the ~65% that have a matching CSV row.

Each marker is placed at its CoTracker-predicted pixel position AT THE TARGET FRAME,
backprojected using the target frame's OWN depth map -- see overlay_all_traces.py's docstring
for why (this guarantees on-surface placement; an earlier version placed markers using each
query's own seed frame + a world_view_transform camera-motion correction, which did not fix
markers rendering behind the reconstruction since the actual cause is tissue deformation
between frames, not camera motion).

Usage:
    python overlay_all_traces_random.py \
        --reconstruct_dir reconstruct/headAndNeck/P202/run7_densified \
        --tracks data/headAndNeck/tracks/P202/run7/cotracker_tracks.npy \
        --seed_frames data/headAndNeck/tracks/P202/run7/cotracker_seed_frames.npy \
        --target_frame 12 \
        --output frame_12_densified_rand_traces_.ply
"""

import os
import random
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from overlay_tracks import backproject_pixel, blend_recolor_nearby, FLIM_COLOR_SCALE


def main():
    parser = ArgumentParser(description="Overlay ALL FLIm queries onto a single frame's point "
                                         "cloud, with a random FLIM_COLOR_SCALE severity bucket "
                                         "(green=healthy..blue=cancer) per marker")
    parser.add_argument("--reconstruct_dir", type=str, required=True,
                        help="Output of `render.py --reconstruct` -- must contain frame_<i>.ply "
                             "plus depths.npy/camera_params.npy/frame_indices.npy/frame_uids.npy")
    parser.add_argument("--tracks", type=str, required=True,
                        help="Per-query CoTracker tracks .npy, shape [T, N, 2], from extract_tracks.py")
    parser.add_argument("--seed_frames", type=str, required=True,
                        help="Per-query seed frame .npy [N] from extract_tracks.py -- each query's "
                             "marker is placed using ITS OWN seed frame's depth/pose")
    parser.add_argument("--target_frame", type=int, default=40,
                        help="ply filename (frame_<target_frame>.ply) whose point cloud all markers "
                             "get stamped onto")
    parser.add_argument("--output", type=str, default="frame_all_traces_random.ply")
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
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed, for reproducible marker color assignment")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    depths = np.load(os.path.join(args.reconstruct_dir, "depths.npy"))
    frame_idxs = np.load(os.path.join(args.reconstruct_dir, "frame_indices.npy"))
    focal_x, focal_y, width, height = np.load(os.path.join(args.reconstruct_dir, "camera_params.npy"))
    width, height = int(width), int(height)
    k_target = int(np.where(frame_idxs == args.target_frame)[0][0])
    target_depth = depths[k_target]

    tracks = np.load(args.tracks)
    assert tracks.ndim == 3 and tracks.shape[2] == 2, f"Expected [T, N, 2], got {tracks.shape}"
    seed_frames = np.load(args.seed_frames)
    T, N, _ = tracks.shape
    assert seed_frames.shape == (N,), f"Expected seed_frames shape ({N},), got {seed_frames.shape}"

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

    n_placed = 0
    n_touched_total = 0
    skip_no_depth = 0

    for i in range(N):
        uv = tracks[args.target_frame, i]
        cam_pt = backproject_pixel(target_depth, uv[0], uv[1], focal_x, focal_y, width, height)
        if cam_pt is None:
            skip_no_depth += 1
            continue

        color = random.choice(FLIM_COLOR_SCALE)

        if tree is not None:
            n_touched_total += blend_recolor_nearby(
                tree, base_colors, original_colors, cam_pt, solid_radius, blend_radius, color,
                ease_power=args.blend_ease_power)
        n_placed += 1

    print(f"Placed {n_placed}/{N} FLIm markers onto frame {args.target_frame} with a random "
          f"FLIM_COLOR_SCALE severity bucket each ({skip_no_depth} skipped: no valid depth at "
          f"the target frame's tracked pixel), {n_touched_total} real points recolored")

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
