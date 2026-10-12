"""
Overlay CoTracker/FLIm 2D tracks onto EndoGaussian's reconstructed point clouds, using
depth-guided back-projection from the model's OWN rendered depth at the tracked pixel -- the
same core formula as the UC Davis FLIm-overlay paper (track sparse FLIm pixels through time,
then for each frame back-project using the rendered depth D̂ at (x,y):
Xc=(x-cx)*d/fx, Yc=(y-cy)*d/fy, Zc=d). Unlike the paper, which injects markers as new Gaussians
into one continuously-evolving world-space model, this repo exports a SEPARATE point cloud per
frame in that frame's own camera space (no shared world frame -- see backproject_pixel's
docstring below), so the marker here is placed in that same per-frame camera-space convention.

Earlier versions of this script found the 3D marker location either by projecting every point
in the exported point cloud into 2D and nearest-neighbor matching to the tracked pixel (a poor
approximation, dependent on local cloud density and wrong near depth discontinuities), or by
reloading the trained GaussianModel and re-rendering depth on demand (correct in principle, but
required a second model load and its own render pass, and re-rendering without also
reproducing `reconstruct_point_cloud`'s alpha/mask/bilateral depth cleaning gave garbage depth
at low-alpha/background pixels).

This version instead reads the EXACT depth map and camera pose that `render.py --reconstruct`
already computed and used to build each frame's own point cloud (now saved alongside it as
depths.npy / world_view_transforms.npy / camera_params.npy / frame_indices.npy -- see
render.py::reconstruct_point_cloud), so the back-projection uses the same cleaned depth the
base cloud itself was built from, with no model reload and no re-derivation of that cleanup.

Every valid FLIm detection is its own independently-tracked CoTracker query (see
extract_tracks.py). For each reconstructed frame, this script places one marker for EVERY query
that has already been seeded by that point in the video (seed_frame <= current frame) and that
CoTracker still considers visible -- i.e. every distinct piece of tissue the FLIm laser has
touched so far, each shown at wherever it has currently drifted to. Markers accumulate over the
video: frame N's output shows every FLIm mark made up through frame N, not just the latest one.

Usage:
    # 1) Produce the base reconstruction + its saved depth/pose (one-time, per scene):
    python render.py --model_path output/headAndNeck/P101/run3_cropped \
        --configs arguments/headAndNeck.py --skip_train --skip_video --reconstruct

    # 2) Overlay tracks onto it:
    python overlay_tracks.py \
        --reconstruct_dir reconstruct/test \
        --tracks cotracker_tracks.npy --visibility cotracker_visibility.npy \
        --seed_frames cotracker_seed_frames.npy \
        --output_dir reconstruct/test_tracked

    --tracks/--visibility/--seed_frames: per-query CoTracker output from extract_tracks.py,
        shapes [T, N, 2] / [T, N] / [N] -- N is the number of distinct FLIm detections tracked,
        T is indexed by frame number in whatever video CoTracker actually ran on (see
        --track_frame_stride/--track_frame_offset if that video isn't the same one this
        reconstruction's frames came from)
"""

import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

HEALTHY_LABELS = [1, 2, 18, 19, 20, 25, 27, 29, 36, 37]
IGNORE_LABELS = [3, 4, 23, 26, 31, 38, 39, 40, 41, 42]
CAUTERY_LABELS = [43, 44, 45, 46]
LGD_LABELS = [21, 32, 34]
HGD_LABELS = [35]
CANCER_LABELS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 24, 28, 30, 33]
LABELS_INDEX = [HEALTHY_LABELS, IGNORE_LABELS, CAUTERY_LABELS, LGD_LABELS, HGD_LABELS, CANCER_LABELS]
SEVERITY_SCALE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# "Green To Blue Gradient" scheme, index-aligned with LABELS_INDEX/SEVERITY_SCALE above: index 0
# (HEALTHY_LABELS, severity 0.0) -> Vivid Green, ... index 5 (CANCER_LABELS, severity 1.0) ->
# Dark Sea Blue. Replaces an earlier single-hue-blue/saturation-ramp encoding, which compressed
# all 6 severity buckets into a brightness/saturation difference on the same hue -- easy to
# misread at a glance, especially the low-saturation severe end next to pinkish tissue. A
# distinct color per bucket (green=cancer-free ... blue=cancer) reads unambiguously regardless
# of which bucket a marker happens to land in.
FLIM_COLOR_SCALE_HEX = ["#00D40E", "#00BA2D", "#009658", "#007185", "#0053A8", "#003193"]


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


FLIM_COLOR_SCALE = [hex_to_rgb01(h) for h in FLIM_COLOR_SCALE_HEX]


def flim_marker_color(df, full_frame):
    """Green-to-blue severity encoding: among all CSV rows for this frame, find the
    highest-severity LABELS_INDEX bucket any WLI_4 value falls into, and look up that bucket's
    color directly from FLIM_COLOR_SCALE (index-aligned with LABELS_INDEX/SEVERITY_SCALE --
    bucket 0=healthy=green ... bucket 5=cancer=blue).

    Returns None if no CSV row matches this frame, or none of its WLI_4 values fall in any known
    bucket -- the caller skips placing a marker entirely for such a query, leaving the underlying
    reconstructed tissue color untouched rather than drawing a marker with no real reading behind it.
    """
    rows = df[np.floor(df["WLI_1"]) == full_frame]
    flim_values = rows["WLI_4"].values
    matched_index = None
    for index, scale in enumerate(LABELS_INDEX):
        for flim_val in flim_values:
            if flim_val in scale and (matched_index is None or index > matched_index):
                matched_index = index
    if matched_index is None:
        return None
    return FLIM_COLOR_SCALE[matched_index]


def blend_recolor_nearby(tree, colors, original_colors, center, radius, blend_radius, marker_color,
                          ease_power=0.4):
    """Mark a FLIm detection by recoloring the REAL nearby base-cloud points, instead of adding
    synthetic points near an estimated surface (a sphere, or a flattened disc along an estimated
    normal). The real points are already exactly on the true tissue surface -- curved, folded,
    whatever shape it actually is -- so there's no approximation error from guessing a local
    normal/tangent plane: recoloring them makes the marker literally conform to the actual
    reconstructed geometry instead of poking through or hovering near it.

    Points within `radius` of `center` (found via `tree`, built over the same cloud `colors`
    indexes into) get set to the exact `marker_color` -- a solid core. Points out to
    `blend_radius` get blended between `marker_color` (at `radius`) and that point's OWN color (at
    `blend_radius`) by distance, so the marker fades into its real surroundings instead of ending
    in a hard color boundary.

    The blend fraction `t` is raised to `ease_power` (< 1) before mixing, instead of interpolating
    RGB linearly by raw distance. A straight linear RGB blend from a vivid, saturated marker color
    toward a pale tissue tone barely LOOKS like it's blending for most of the distance -- e.g.
    vivid green (0, 0.83, 0.05) mixed 60% of the way to a typical pale tissue tone (0.85, 0.72,
    0.68) is still (0.51, 0.76, 0.43), clearly green-dominant, just dimmer -- because the green
    channel is already close between the two colors, so only red/blue visibly climb, and only
    very near the outer edge does the mix actually read as tissue-colored. Easing `t` with a power
    < 1 front-loads the transition (reaches a visibly tissue-leaning mix much closer to `radius`
    instead of saving it all for right before `blend_radius`), so the fade is visible throughout
    the halo rather than looking like a uniformly dim solid core.

    `original_colors` must be a snapshot of `colors` taken BEFORE any marker in this batch was
    processed -- blend targets are always read from that snapshot, never from the live `colors`
    array, so markers whose blend radii overlap don't compound onto each other.

    Returns the number of real points touched (0 if none fall within blend_radius -- e.g. a
    sparse/thin region of the reconstruction).
    """
    idxs = tree.query_ball_point(center, blend_radius)
    if not idxs:
        return 0
    idxs = np.asarray(idxs)
    dists = np.linalg.norm(tree.data[idxs] - center, axis=1)

    solid = dists <= radius
    colors[idxs[solid]] = marker_color

    blend_mask = ~solid
    if blend_mask.any() and blend_radius > radius:
        b_idx = idxs[blend_mask]
        t_raw = np.clip((dists[blend_mask] - radius) / (blend_radius - radius), 0.0, 1.0)
        t = (t_raw ** ease_power)[:, None]
        colors[b_idx] = np.asarray(marker_color)[None, :] * (1 - t) + original_colors[b_idx] * t

    return len(idxs)


def backproject_pixel(depth_map, u, v, focal_x, focal_y, width, height):
    """depth_map: (H,W) cleaned metric depth (same one used to build the base ply). Returns a
    (3,) point in THAT FRAME'S OWN camera-space coordinates, or None if (u,v) is out of bounds
    or has no valid depth (e.g. masked/background).

    NOTE: each frame_<i>.ply this repo exports is built via Open3D's create_from_rgbd_image,
    which unprojects directly into that frame's own camera space -- it is never transformed
    into a shared world frame (world_view_transform is only used transiently, inside
    render.py's hole-filling, to reproject one frame's points into another's camera view for
    filling gaps; it never touches what actually gets written to disk). So annotating a given
    frame's ply must stay in that same camera-space convention -- no world transform here.

    ALSO NOTE: render.py builds that RGBDImage via `create_from_color_and_depth(rgb_im,
    depth_im, convert_rgb_to_intensity=False)` without passing depth_scale, so Open3D applies
    its default depth_scale=1000.0 (i.e. it divides the raw depth_np values by 1000 before
    unprojecting) -- confirmed empirically: a raw depth value of ~0.485 here corresponds to an
    actual point at z~0.000485 in the exported ply. We have to divide by the same 1000 to land
    in the same units as the cloud we're annotating.
    """
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or ui >= width or vi < 0 or vi >= height:
        return None
    d = float(depth_map[vi, ui]) / 1000.0
    if d <= 0:
        return None

    cx, cy = width / 2.0, height / 2.0
    x_cam = (u - cx) * d / focal_x
    y_cam = (v - cy) * d / focal_y
    return np.array([x_cam, y_cam, d])


def reproject_across_frames(depth_map, u, v, wvt_source, wvt_target, focal_x, focal_y, width, height):
    """Like backproject_pixel, but for placing a marker computed at pixel (u,v) of the SOURCE
    frame's OWN depth map onto a DIFFERENT TARGET frame's point cloud. Naively reusing
    backproject_pixel's camera-space output for that (as overlay_all_traces.py first did) is
    wrong whenever the camera moved between the two frames -- "in front" in one frame's camera
    space is not "in front" in another's. This instead: (1) backprojects in RAW depth units
    (matching render.py's own _unproject_to_world/_reproject_from_world, which is what its
    world_view_transforms.npy values are meant to be used with -- NOT the /1000-scaled units
    backproject_pixel returns), (2) transforms world-ward via the SOURCE frame's
    world_view_transform (world->camera, row-vector: p_cam_row = p_world_row_homog @ wvt), (3)
    transforms into the TARGET frame's camera space via the TARGET's world_view_transform, then
    (4) divides by 1000 at the end to land in the same ply-scale units backproject_pixel uses.

    Returns a (3,) point in the TARGET frame's OWN camera-space (ply-scale) coordinates, or
    None if (u,v) is out of bounds/has no valid depth in the source frame, or the point
    reprojects behind the target camera.
    """
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or ui >= width or vi < 0 or vi >= height:
        return None
    d = float(depth_map[vi, ui])
    if d <= 0:
        return None

    cx, cy = width / 2.0, height / 2.0
    x_cam = (ui - cx) * d / focal_x
    y_cam = (vi - cy) * d / focal_y
    cam_pt_h = np.array([x_cam, y_cam, d, 1.0])
    world_pt_h = cam_pt_h @ np.linalg.inv(wvt_source)
    target_pt_h = world_pt_h @ wvt_target
    x_t, y_t, z_t = target_pt_h[:3]
    if z_t <= 1e-8:
        return None
    return np.array([x_t, y_t, z_t]) / 1000.0


def main():
    parser = ArgumentParser(description="Overlay CoTracker/FLIm tracks on reconstructed "
                                         "point clouds using the model's own rendered depth")
    parser.add_argument("--reconstruct_dir", type=str, required=True,
                        help="Output of `render.py --reconstruct` -- must contain frame_<i>.ply "
                             "plus depths.npy/camera_params.npy/frame_indices.npy/frame_uids.npy")
    parser.add_argument("--tracks", type=str, required=True,
                        help="Per-query CoTracker tracks .npy file, shape [T, N, 2], from "
                             "extract_tracks.py. T is indexed by frame number in whatever video "
                             "CoTracker actually ran on (see --track_frame_stride/"
                             "--track_frame_offset if that video isn't the same one this "
                             "reconstruction's frames came from)")
    parser.add_argument("--visibility", type=str, default="",
                        help="Per-query visibility mask .npy [T, N] from extract_tracks.py")
    parser.add_argument("--seed_frames", type=str, required=True,
                        help="Per-query seed frame .npy [N] from extract_tracks.py -- query i is "
                             "only placed on frames whose (converted) index is >= seed_frames[i]")
    parser.add_argument("--track_frame_stride", type=int, default=1,
                        help="Set this when the reconstruction's source images are a subsampled "
                             "version of the video CoTracker actually tracked on -- e.g. "
                             "run3_half keeps every 5th frame of the original clip, so its own "
                             "local frame index k corresponds to original/tracked frame k*5. "
                             "tracks/visibility/seed_frames are indexed by "
                             "frame_uids[k]*stride + offset.")
    parser.add_argument("--track_frame_offset", type=int, default=0,
                        help="See --track_frame_stride.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory (default: reconstruct_dir + '_tracked')")
    parser.add_argument("--marker_color", type=int, nargs=3, default=[0, 255, 0],
                        help="Marker color RGB 0-255 (default: green)")
    parser.add_argument("--marker_radius", type=float, default=None,
                        help="Reference radius for a marker's overall size, in the same units "
                             "as the point cloud -- see --solid_core_fraction/--blend_radius_"
                             "factor for how it's actually split into a solid core vs. a blend "
                             "halo. Default: auto-scaled per-frame to 0.22%% of that frame's own "
                             "bounding-box diagonal -- these clouds are built with Open3D's "
                             "implicit depth_scale=1000 (see backproject_pixel), so their scale "
                             "is typically sub-millimeter and a fixed absolute default like the "
                             "old 0.005 would dwarf the entire reconstruction.")
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
                             "before mixing colors (<1 front-loads the transition). A plain "
                             "linear RGB blend (power=1.0) from a vivid marker color toward a "
                             "pale tissue tone barely looks like it's fading for most of the "
                             "halo -- e.g. green mixed 60%% of the way to tissue is still "
                             "clearly green, just dimmer -- because the green channel is "
                             "already close between the two colors, so only red/blue visibly "
                             "shift. A power < 1 reaches a visibly tissue-leaning mix much "
                             "sooner instead of saving nearly all the change for right before "
                             "the halo's outer edge. See blend_recolor_nearby.")
    parser.add_argument("--color_by_flim", action="store_true",
                        help="Color each marker by the WLI_4 tissue-label severity found at "
                             "its frame in the FLIm CSV, instead of a fixed --marker_color. "
                             "Uses the Green To Blue Gradient scheme (FLIM_COLOR_SCALE): a "
                             "distinct color per severity bucket, green (healthy) through blue "
                             "(cancer) -- see flim_marker_color. Queries with no matching/"
                             "known-label CSV row at "
                             "their seed frame are never marked at all -- the underlying "
                             "reconstructed tissue color is left untouched for them.")
    parser.add_argument("--flim_csv", type=str,
                        default="/media/data/zliwang/data/HN_P2C_20250304.csv",
                        help="FLIm CSV path (only used with --color_by_flim)")
    parser.add_argument("--csv_subject_id", type=int, default=None,
                        help="SubjectID to filter the CSV to (required with --color_by_flim)")
    parser.add_argument("--csv_run", type=int, default=None,
                        help="Run number to filter the CSV to (required with --color_by_flim)")
    parser.add_argument("--csv_frame_stride", type=int, default=1,
                        help="The CSV's WLI_1 column is indexed to the FULL original video "
                             "(0-based, one row group per video frame). Each query's color is "
                             "fixed once, from the WLI_4 label at its OWN seed frame (the frame "
                             "the FLIm device actually registered that detection) -- converted "
                             "via seed_frames[i]*csv_frame_stride + csv_frame_offset, where "
                             "seed_frames is in the same frame space as the tracked video (see "
                             "--track_frame_stride). E.g. for run3_half (every 5th frame of the "
                             "full run3 video, confirmed by matching frame content), pass "
                             "--csv_frame_stride 5.")
    parser.add_argument("--csv_frame_offset", type=int, default=0,
                        help="See --csv_frame_stride.")
    args = parser.parse_args()

    depths = np.load(os.path.join(args.reconstruct_dir, "depths.npy"))
    frame_idxs = np.load(os.path.join(args.reconstruct_dir, "frame_indices.npy"))   # ply filename
    frame_uids = np.load(os.path.join(args.reconstruct_dir, "frame_uids.npy"))      # true video frame #
    focal_x, focal_y, width, height = np.load(os.path.join(args.reconstruct_dir, "camera_params.npy"))
    width, height = int(width), int(height)

    tracks = np.load(args.tracks)
    assert tracks.ndim == 3 and tracks.shape[2] == 2, f"Expected [T, N, 2], got {tracks.shape}"
    visibility = np.load(args.visibility) if args.visibility else None
    seed_frames = np.load(args.seed_frames)
    T, N, _ = tracks.shape
    assert seed_frames.shape == (N,), f"Expected seed_frames shape ({N},), got {seed_frames.shape}"
    if visibility is not None:
        assert visibility.shape == (T, N), f"Expected visibility shape ({T}, {N}), got {visibility.shape}"

    output_dir = args.output_dir or (args.reconstruct_dir.rstrip('/') + '_tracked')
    os.makedirs(output_dir, exist_ok=True)

    fixed_color_float = np.array(args.marker_color) / 255.0

    flim_df = None
    if args.color_by_flim:
        assert args.csv_subject_id is not None and args.csv_run is not None, \
            "--color_by_flim requires --csv_subject_id and --csv_run"
        flim_df = pd.read_csv(args.flim_csv, low_memory=False)
        flim_df = flim_df[(flim_df["SubjectID"] == args.csv_subject_id) &
                           (flim_df["Run"] == args.csv_run)]
        print(f"  --color_by_flim: loaded {len(flim_df)} CSV rows for "
              f"SubjectID={args.csv_subject_id} Run={args.csv_run}")

    # Each query's color is fixed once, from the WLI_4 label at its OWN seed frame -- the
    # frame the FLIm device actually registered that detection -- not from whatever frame it's
    # currently being displayed on as it drifts with the tissue. Queries with no matching CSV
    # label (query_has_color[i] == False) never get a marker at all -- see flim_marker_color.
    query_colors = np.zeros((N, 3), dtype=np.float64)
    if flim_df is not None:
        query_has_color = np.zeros(N, dtype=bool)
        for i in range(N):
            full_frame = int(seed_frames[i]) * args.csv_frame_stride + args.csv_frame_offset
            color = flim_marker_color(flim_df, full_frame)
            if color is not None:
                query_colors[i] = color
                query_has_color[i] = True
        print(f"  --color_by_flim: {query_has_color.sum()}/{N} queries matched a known label "
              f"(the rest will never be marked -- no known FLIm reading at their seed frame)")
    else:
        query_colors[:] = fixed_color_float
        query_has_color = np.ones(N, dtype=bool)

    n_frames_annotated = 0
    n_markers_total = 0
    skip_reasons = {"out_of_range": 0, "no_active_queries": 0, "no_valid_depth": 0}
    print(f"Processing {len(frame_idxs)} reconstructed frames "
          f"(camera: fx={focal_x:.2f}, fy={focal_y:.2f}, {width}x{height})")
    print(f"  {N} tracked FLIm queries, tracks array length T={T}, true_frame range in this "
          f"reconstruction: {int(frame_uids.min())}-{int(frame_uids.max())}")

    for k, i_frame in enumerate(frame_idxs):
        i_frame = int(i_frame)         # ply filename position
        # frame_uids[k] is this reconstruction's own local frame number; convert to whatever
        # frame number the tracks array is indexed by (identical unless the reconstruction's
        # source images are a subsampled version of the tracked video -- see --track_frame_stride)
        true_frame = int(frame_uids[k]) * args.track_frame_stride + args.track_frame_offset
        ply_path = os.path.join(args.reconstruct_dir, f"frame_{i_frame}.ply")
        out_path = os.path.join(output_dir, f"frame_{i_frame}.ply")
        pcd = o3d.io.read_point_cloud(ply_path)

        marker_pts = []
        marker_cols = []
        if true_frame >= T:
            skip_reasons["out_of_range"] += 1
        else:
            # Every query seeded by this point in the video and still visible here gets its
            # own marker at its own currently-tracked position -- markers accumulate over time.
            active = np.where((seed_frames <= true_frame) &
                               (visibility[true_frame] if visibility is not None else True) &
                               query_has_color)[0]
            if len(active) == 0:
                skip_reasons["no_active_queries"] += 1
            for i in active:
                uv = tracks[true_frame, i]
                cam_pt = backproject_pixel(depths[k], uv[0], uv[1], focal_x, focal_y, width, height)
                if cam_pt is None:
                    skip_reasons["no_valid_depth"] += 1
                    continue
                marker_pts.append(cam_pt)
                marker_cols.append(query_colors[i])

        if len(marker_pts) == 0:
            o3d.io.write_point_cloud(out_path, pcd)
            continue

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        if args.marker_radius is not None:
            radius = args.marker_radius
        elif len(points) > 0:
            radius = (0.02 / 9.0) * np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        else:
            radius = 0.005

        solid_radius = radius * args.solid_core_fraction
        blend_radius = radius * args.blend_radius_factor
        original_colors = colors.copy()
        tree = cKDTree(points) if len(points) > 0 else None
        n_touched = 0
        for cam_pt, marker_color_float in zip(marker_pts, marker_cols):
            if tree is not None:
                n_touched += blend_recolor_nearby(
                    tree, colors, original_colors, cam_pt, solid_radius, blend_radius, marker_color_float,
                    ease_power=args.blend_ease_power)

        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(out_path, pcd)
        n_frames_annotated += 1
        n_markers_total += len(marker_pts)

        if k % 50 == 0:
            print(f"  ply frame_{i_frame} (true video frame {true_frame}): "
                  f"{len(marker_pts)} active FLIm markers, {n_touched} real points recolored")

    print(f"\nDone. Annotated {n_frames_annotated}/{len(frame_idxs)} frames with "
          f"{n_markers_total} total markers placed. Saved to: {output_dir}")
    if n_frames_annotated < len(frame_idxs):
        print(f"Frames with zero markers: "
              f"{skip_reasons['out_of_range']} true_frame >= T (tracks array), "
              f"{skip_reasons['no_active_queries']} had no seeded-and-visible query yet")
    if skip_reasons["no_valid_depth"] > 0:
        print(f"{skip_reasons['no_valid_depth']} individual marker placements skipped "
              f"(no valid depth at the tracked pixel -- out of image bounds, or "
              f"masked/background there)")


if __name__ == "__main__":
    main()
