"""
Overlay CoTracker 2D tracks onto EndoGaussian's reconstructed point clouds.

For each frame:
  1. Loads the point cloud PLY from EndoGaussian's --reconstruct output
  2. Takes the CoTracker 2D pixel coordinate for that frame
  3. Unprojects to 3D using the same depth/intrinsics used to build the cloud
  4. Finds the nearest points in the cloud and recolors them
  5. Adds a cluster of marker points (larger visible blob) at the tracked location
  6. Saves the annotated PLY

Usage:
    python overlay_tracks.py \
        --reconstruct_dir reconstruct/test \
        --tracks cotracker_tracks.npy \
        --camera_params 332.0 332.0 320 256 \
        --output_dir reconstruct/test_tracked

    --tracks: CoTracker output, shape [T, 2] or [T, N, 2] (N tracked points)
    --camera_params: fx fy width height (same as used in reconstruction)
    --marker_color: R G B (0-255), default bright green
    --marker_radius: size of the synthetic marker ball (in world units)
    --num_neighbors: how many nearby cloud points to recolor
"""

import numpy as np
import open3d as o3d
import os
from argparse import ArgumentParser
from scipy.spatial import cKDTree


def unproject_pixel(u, v, depth, fx, fy, cx, cy):
    """Unproject pixel (u,v) with depth to 3D camera-space coordinates.

    Note: Open3D's create_from_rgbd_image uses the same unprojection,
    so this gives us a point that should exist in the cloud.
    """
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z])


def create_marker_sphere(center, radius, color, num_points=50):
    """Create a small sphere of colored points as a visible marker."""
    points = []
    for _ in range(num_points):
        offset = np.random.randn(3)
        offset = offset / np.linalg.norm(offset) * radius * np.random.uniform(0, 1)
        points.append(center + offset)
    points = np.array(points)
    colors = np.tile(color, (num_points, 1))
    return points, colors


def annotate_point_cloud(pcd_path, track_uv, fx, fy, width, height,
                         marker_color, marker_radius, num_neighbors,
                         marker_points):
    """Load a point cloud, find the tracked point, annotate and return it."""
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        return pcd

    cx = width / 2.0
    cy = height / 2.0

    # The point cloud was built with create_from_rgbd_image which unprojects
    # each pixel to camera space. We do the same for our tracked pixel.
    # But we need the depth at that pixel — we get it by finding the nearest
    # point in the cloud to the ray through (u, v).

    u, v = track_uv[0], track_uv[1]

    # Method: find the point whose projected pixel is closest to (u, v)
    # Project all cloud points back to 2D
    z_vals = points[:, 2]
    valid = z_vals > 0
    if not valid.any():
        return pcd

    proj_u = (points[valid, 0] * fx / points[valid, 2]) + cx
    proj_v = (points[valid, 1] * fy / points[valid, 2]) + cy
    proj_pixels = np.stack([proj_u, proj_v], axis=1)

    # Find nearest projected point to our target pixel
    dists_2d = np.sqrt((proj_pixels[:, 0] - u) ** 2 + (proj_pixels[:, 1] - v) ** 2)
    nearest_idx_in_valid = np.argmin(dists_2d)

    # Map back to full point array index
    valid_indices = np.where(valid)[0]
    center_idx = valid_indices[nearest_idx_in_valid]
    center_3d = points[center_idx]

    pixel_dist = dists_2d[nearest_idx_in_valid]
    if pixel_dist > 20:
        print(f"  WARNING: nearest point is {pixel_dist:.1f}px away from target")

    # Find K nearest neighbors in 3D around the center point and recolor them
    tree = cKDTree(points)
    _, neighbor_indices = tree.query(center_3d, k=num_neighbors)
    marker_color_float = np.array(marker_color) / 255.0
    for idx in neighbor_indices:
        colors[idx] = marker_color_float

    # Create synthetic marker sphere (larger visible blob)
    sphere_pts, sphere_colors = create_marker_sphere(
        center_3d, marker_radius, marker_color_float, num_points=marker_points
    )

    # Combine original cloud + marker sphere
    all_points = np.vstack([points, sphere_pts])
    all_colors = np.vstack([colors, sphere_colors])

    annotated = o3d.geometry.PointCloud()
    annotated.points = o3d.utility.Vector3dVector(all_points)
    annotated.colors = o3d.utility.Vector3dVector(all_colors)

    return annotated


def main():
    parser = ArgumentParser(description="Overlay CoTracker tracks on reconstructed point clouds")
    parser.add_argument("--reconstruct_dir", type=str, required=True,
                        help="Directory containing frame_0.ply, frame_1.ply, ...")
    parser.add_argument("--tracks", type=str, required=True,
                        help="CoTracker tracks .npy file, shape [T, 2] or [T, N, 2]")
    parser.add_argument("--camera_params", type=float, nargs=4, required=True,
                        help="fx fy width height (same intrinsics used for reconstruction)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory (default: reconstruct_dir + '_tracked')")
    parser.add_argument("--marker_color", type=int, nargs=3, default=[0, 255, 0],
                        help="Marker color RGB 0-255 (default: green)")
    parser.add_argument("--marker_radius", type=float, default=0.005,
                        help="Radius of the marker sphere in world units")
    parser.add_argument("--num_neighbors", type=int, default=20,
                        help="Number of nearby points to recolor")
    parser.add_argument("--marker_points", type=int, default=80,
                        help="Number of synthetic points in the marker sphere")
    parser.add_argument("--track_index", type=int, default=0,
                        help="If tracks has multiple points [T, N, 2], which one to use")
    parser.add_argument("--visibility", type=str, default="",
                        help="Optional visibility mask .npy [T] from CoTracker")

    args = parser.parse_args()

    fx, fy, width, height = args.camera_params
    width, height = int(width), int(height)

    # Load tracks
    tracks = np.load(args.tracks)
    print(f"Loaded tracks: shape {tracks.shape}")
    if tracks.ndim == 3:
        tracks = tracks[:, args.track_index, :]  # [T, 2]
    assert tracks.ndim == 2 and tracks.shape[1] == 2, \
        f"Expected tracks shape [T, 2], got {tracks.shape}"

    # Load visibility if provided
    visibility = None
    if args.visibility:
        visibility = np.load(args.visibility)
        print(f"Loaded visibility: shape {visibility.shape}")

    # Output directory
    output_dir = args.output_dir or (args.reconstruct_dir.rstrip('/') + '_tracked')
    os.makedirs(output_dir, exist_ok=True)

    # Find all frame PLY files
    ply_files = sorted([f for f in os.listdir(args.reconstruct_dir) if f.endswith('.ply')])
    print(f"Found {len(ply_files)} point cloud frames")

    num_frames = min(len(ply_files), len(tracks))
    print(f"Processing {num_frames} frames...")
    print(f"  Camera: fx={fx}, fy={fy}, {width}x{height}")
    print(f"  Marker: color={args.marker_color}, radius={args.marker_radius}, "
          f"neighbors={args.num_neighbors}")

    for i in range(num_frames):
        ply_path = os.path.join(args.reconstruct_dir, ply_files[i])
        uv = tracks[i]

        # Skip if point is not visible
        if visibility is not None and not visibility[i]:
            # Just copy the original without annotation
            pcd = o3d.io.read_point_cloud(ply_path)
            out_path = os.path.join(output_dir, ply_files[i])
            o3d.io.write_point_cloud(out_path, pcd)
            continue

        # Skip if pixel is outside image bounds
        if uv[0] < 0 or uv[0] >= width or uv[1] < 0 or uv[1] >= height:
            pcd = o3d.io.read_point_cloud(ply_path)
            out_path = os.path.join(output_dir, ply_files[i])
            o3d.io.write_point_cloud(out_path, pcd)
            continue

        annotated = annotate_point_cloud(
            ply_path, uv, fx, fy, width, height,
            args.marker_color, args.marker_radius,
            args.num_neighbors, args.marker_points
        )

        out_path = os.path.join(output_dir, ply_files[i])
        o3d.io.write_point_cloud(out_path, annotated)

        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}: pixel ({uv[0]:.0f}, {uv[1]:.0f})")

    print(f"\nDone. Annotated point clouds saved to: {output_dir}")


if __name__ == "__main__":
    main()
