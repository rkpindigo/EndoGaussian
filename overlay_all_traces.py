"""
Overlay ALL FLIM trace points onto a single frame's point cloud.

Shows the complete laser trace path on one reconstruction frame.
Each trace point is a small colored marker — no neighbor recoloring.

Usage:
    python overlay_all_traces.py \
        --reconstruct_dir reconstruct/test \
        --tracks flim_tracks_pixels.npy \
        --visibility flim_visibility.npy \
        --camera_params 1536.0 1536.0 1280 720 \
        --target_frame 400 \
        --output frame_400_all_traces.ply
"""

import numpy as np
import open3d as o3d
import os
from argparse import ArgumentParser
from scipy.spatial import cKDTree
import pandas as pd
import colorsys
import math

HEALTHY_LABELS = [1, 2, 18, 19, 20, 25, 27, 29, 36, 37]
IGNORE_LABELS = [3, 4, 23, 26, 31, 38, 39, 40, 41, 42]
CAUTERY_LABELS = [43, 44, 45, 46]
LGD_LABELS = [21, 32, 34]
HGD_LABELS = [35]
CANCER_LABELS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 24, 28, 30, 33]
LABELS_INDEX = [HEALTHY_LABELS, IGNORE_LABELS, CAUTERY_LABELS, LGD_LABELS, HGD_LABELS, CANCER_LABELS]
SATURATION_SCALE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def hsb_to_rgb(h, s, v):
    """
    Converts HSV to RGB (0.0 - 1.0).
    h: 0.0 - 1.0 (e.g. 0.33 for Green, 0.0 for Red, 0.66 for Blue)
    s: 0.0 - 1.0 (0.0 = completely grey/grayscale, 1.0 = hyper-saturated)
    v: 0.0 - 1.0 (0.0 = black, 1.0 = maximum brightness)
    """
    r, g, v_val = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g, v_val])

def find_3d_for_pixel(pcd_points, u, v, fx, fy, cx, cy):
    """Find the 3D point in the cloud that corresponds to pixel (u, v)."""
    z_vals = pcd_points[:, 2]
    valid = z_vals > 0
    if not valid.any():
        return None

    proj_u = (pcd_points[valid, 0] * fx / pcd_points[valid, 2]) + cx
    proj_v = (pcd_points[valid, 1] * fy / pcd_points[valid, 2]) + cy

    dists_2d = (proj_u - u) ** 2 + (proj_v - v) ** 2
    nearest_idx = np.argmin(dists_2d)
    min_dist = np.sqrt(dists_2d[nearest_idx])

    if min_dist > 150:
        return None

    valid_indices = np.where(valid)[0]
    return pcd_points[valid_indices[nearest_idx]]


def create_marker(center, radius, color, num_points=20):
    """Create a small cluster of points as a marker."""
    points = []
    for _ in range(num_points):
        offset = np.random.randn(3)
        offset = offset / np.linalg.norm(offset) * radius * np.random.uniform(0, 1)
        points.append(center + offset)
    return np.array(points), np.tile(color, (num_points, 1))


def main():
    parser = ArgumentParser(description="Overlay all FLIM trace points on one frame")
    parser.add_argument("--reconstruct_dir", type=str, required=True)
    parser.add_argument("--tracks", type=str, required=True)
    parser.add_argument("--visibility", type=str, default="")
    parser.add_argument("--camera_params", type=float, nargs=4, required=True,
                        help="fx fy width height")
    parser.add_argument("--target_frame", type=int, default=400,
                        help="Which frame's point cloud to use as the base")
    parser.add_argument("--output", type=str, default="frame_all_traces.ply")
    parser.add_argument("--marker_color", type=int, nargs=3, default=[149, 84, 74],
                        help="Marker color RGB 0-255")
    parser.add_argument("--marker_radius", type=float, default=0.001,
                        help="Small radius for each trace point")
    parser.add_argument("--marker_points", type=int, default=15,
                        help="Points per marker (smaller = less prominent)")

    args = parser.parse_args()

    #read the full CSV file
    print("Reading CSV file...")

    #load csv file and access rows where patient and run match what's currently being run
    #get WLI_4 and frame columns
    df = pd.read_csv("/media/data/zliwang/data/HN_P2C_20250304.csv")

    columns = [
        "WLI_1", #original frame index where the FLIm data was collected
        "Run",
        "WLI_4",
    ]

    df = df[(df["SubjectID"] == 101) & (df["Run"] == 3)]

    df = df[columns]

    print("\nNumber of rows after filtering: ", len(df))

    fx, fy, width, height = args.camera_params
    width, height = int(width), int(height)
    cx, cy = width / 2.0, height / 2.0

    tracks = np.load(args.tracks)
    print(f"Loaded tracks: {tracks.shape}")

    visibility = None
    if args.visibility:
        visibility = np.load(args.visibility)

    #find all PLY files
    ply_files = sorted([f for f in os.listdir(args.reconstruct_dir) if f.endswith('.ply')])
    print(f"Found {len(ply_files)} point cloud frames")

    #load target frame as base
    target_ply = os.path.join(args.reconstruct_dir, f'frame_{args.target_frame}.ply')
    if not os.path.exists(target_ply):
        target_ply = os.path.join(args.reconstruct_dir, ply_files[args.target_frame])
    base_pcd = o3d.io.read_point_cloud(target_ply)
    base_points = np.asarray(base_pcd.points)
    base_colors = np.asarray(base_pcd.colors)
    print(f"Base frame {args.target_frame}: {len(base_points)} points")

    #for each visible frame, find 3D position of FLIM mark from said frame's own point cloud, then place it on base    
    h, s, v = colorsys.rgb_to_hsv(args.marker_color[0] / 255.0, args.marker_color[1] / 255.0, args.marker_color[2] / 255.0)
    
    all_marker_points = []
    all_marker_colors = []
    found_count = 0


    for index, scale in enumerate(LABELS_INDEX):
        print(scale)

    num_frames = min(len(ply_files), len(tracks))
    for i in range(num_frames):
        if visibility is not None and not visibility[i]:
            continue

        uv = tracks[i]
        if uv[0] == 0 and uv[1] == 0:
            continue
        if uv[0] < 0 or uv[0] >= width or uv[1] < 0 or uv[1] >= height:
            continue

        # Load this frame's point cloud to get the 3D position
        frame_ply = os.path.join(args.reconstruct_dir, ply_files[i])
        frame_pcd = o3d.io.read_point_cloud(frame_ply)
        frame_points = np.asarray(frame_pcd.points)

        pt_3d = find_3d_for_pixel(frame_points, uv[0], uv[1], fx, fy, cx, cy)
        if pt_3d is None:
            continue

        #take max value of cancer detection index detected
        max_cancer_detection_index = 0
        splice = df[np.floor(df["WLI_1"]) == i]
        flim_values = splice["WLI_4"].values
        for index, scale in enumerate(LABELS_INDEX):
            #go through slice of WLI_4 and frame columns until frame > i
            for flim_val in flim_values:
                if flim_val in scale and index > max_cancer_detection_index:
                    print(f"Frame {i} - Matched flim_val: {flim_val} at index {index}")
                    max_cancer_detection_index = index
        
        hue = SATURATION_SCALE[max_cancer_detection_index] / 3.0 #trying out hue change instead of saturation for now, red to green
        saturation = 1.0
        brightness = 1.0     #brightness

        marker_color_rgb = hsb_to_rgb(hue, saturation, brightness)

        #create a small marker at this 3D position
        m_pts, m_cols = create_marker(pt_3d, args.marker_radius,
                                      marker_color_rgb, args.marker_points)
        all_marker_points.append(m_pts)
        all_marker_colors.append(m_cols)
        found_count += 1

    print(f"Placed {found_count} trace markers out of {num_frames} frames")

    if found_count == 0:
        print("No markers placed. Check tracks and point clouds.")
        return

    # Combine base point cloud + all markers
    marker_points_arr = np.vstack(all_marker_points)
    marker_colors_arr = np.vstack(all_marker_colors)

    combined_points = np.vstack([base_points, marker_points_arr])
    combined_colors = np.vstack([base_colors, marker_colors_arr])

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(combined_points)
    result.colors = o3d.utility.Vector3dVector(combined_colors)

    o3d.io.write_point_cloud(args.output, result)
    print(f"Saved to: {args.output}")
    print(f"Total points: {len(combined_points)} "
          f"({len(base_points)} base + {len(marker_points_arr)} markers)")


if __name__ == "__main__":
    main()
