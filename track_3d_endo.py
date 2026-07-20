import os
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree

def load_xyz_from_ply(ply_path):
    """Extracts just the (x, y, z) coordinates from a 3DGS .ply file."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Could not find .ply file at: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    # Extract x, y, z fields as a numpy array
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    return xyz

def unproject_pixel_to_3d(u, v, depth, intrinsics, c2w):
    """
    Unprojects a 2D pixel coordinate (u, v) using its local depth 
    into a world-space 3D coordinate.
    """
    #cast all parameters to standard float scalars
    u = float(u)
    v = float(v)
    depth = float(depth)
    
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    
    # Step 1: Compute coordinates in camera space
    Z_c = depth
    X_c = (u - cx) * Z_c / fx
    Y_c = (v - cy) * Z_c / fy
    P_camera = np.array([X_c, Y_c, Z_c, 1.0], dtype=np.float32) #force type and flatten
    
    # Step 2: Transform camera space to world space
    P_world = (c2w @ P_camera)[:3]
    return P_world.astype(np.float32).flatten() #force clean 1D shape (3,)

def get_anchor_gaussians(target_3d_pt, canonical_xyz, k_neighbors=4):
    """
    Finds the closest K canonical Gaussians to serve as physical tracking anchors,
    and returns their indices and normalized inverse-distance weights.
    """
    print(f"Finding closest {k_neighbors} anchor Gaussians to {target_3d_pt}...")
    tree = KDTree(canonical_xyz)
    distances, indices = tree.query(target_3d_pt, k=k_neighbors)
    
    # Calculate inverse-distance weights (smooth interpolation)
    # Add epsilon to prevent division-by-zero if exact match
    weights = 1.0 / (distances + 1e-6)
    weights /= weights.sum()
    
    return indices, weights

if __name__ == "__main__":
    COTRACK_FILE = "cotracker_tracks.npy"
    DEPTH_MAP_T0 = "/media/storage1/rpatro/datasets/HeadAndNeck/P202/run7/depth_dam/frame_000001.npy"        # Rendered depth map at t=0
    PLY_DIR = "/media/storage1/rpatro/repos/EndoGaussian/reconstruct/video"           # Folder with frame_0.ply, etc.
    
    # Camera Intrinsics (Adjust fx, fy, cx, cy to match your training set)
    # Double check these inside your EndoGaussian dataset config!
    CAMERA_INTRINSICS = np.array([
        [600.0,   0.0, 320.0],
        [  0.0, 600.0, 256.0],
        [  0.0,   0.0,   1.0]
    ])
    
    # Extrinsic matrix (Camera-to-World) for frame 0 (4x4 matrix)
    C2W_FRAME_0 = np.eye(4) # Replace with your actual 4x4 matrix at frame 0
    # =========================================================================

    # 1. Load the 2D pixel coordinates from your modified CoTracker script
    tracks = np.load(COTRACK_FILE) # Shape [T, 2]
    u0, v0 = float(tracks[0, 0]), float(tracks[0, 1]) #coordinate at frame 0(explicitly cast to scalar floats)
    num_frames = len(tracks)
    print(f"Loaded {num_frames} frames of tracking data. Starting point: ({u0:.1f}, {v0:.1f})")

    # 2. Get depth at start pixel
    depth_map_0 = np.load(DEPTH_MAP_T0)
    print(f"Loaded depth map with shape: {depth_map_0.shape}")

    # --- Robust Depth Resolution ---
    # Case A: If shape is [H, W, C] (like [512, 640, 3] or [512, 640, 1])
    if len(depth_map_0.shape) == 3:
        if depth_map_0.shape[0] in [1, 3]: # Shape is [C, H, W]
            val = depth_map_0[0, int(v0), int(u0)]
        else: # Shape is [H, W, C]
            val = depth_map_0[int(v0), int(u0), 0]
            
    # Case B: If shape is [1, 1, H, W] or similar nested layout
    elif len(depth_map_0.shape) > 3:
        # Squeeze out single-dimensional outer dimensions
        squeezed_depth = np.squeeze(depth_map_0)
        print(f"Squeezed depth map shape to: {squeezed_depth.shape}")
        if len(squeezed_depth.shape) == 3:
            val = squeezed_depth[int(v0), int(u0), 0]
        else:
            val = squeezed_depth[int(v0), int(u0)]
            
    # Case C: Standard [H, W]
    else:
        val = depth_map_0[int(v0), int(u0)]

    # Now convert the resolved single-element slice safely to a float
    d0 = float(np.asarray(val).flat[0])
    print(f"Successfully resolved depth at starting pixel: {d0:.4f}")

    # 3. Correctly calculate the actual unprojected 3D coordinate (P_world_0)
    P_world_0 = unproject_pixel_to_3d(u0, v0, d0, CAMERA_INTRINSICS, C2W_FRAME_0)
    print(f"Unprojected starting coordinate in World Space: {P_world_0}")

    # 4. Load canonical (frame 0) Point Cloud to find spatial anchors
    ply_t0_path = os.path.join(PLY_DIR, "frame_0.ply")
    xyz_t0 = load_xyz_from_ply(ply_t0_path)
    
    # Find nearest Gaussians
    anchor_indices, weights = get_anchor_gaussians(P_world_0, xyz_t0, k_neighbors=4)
    print(f"Locked onto Anchor Indices: {anchor_indices}")
    print(f"Anchor Weights: {weights}")

    # 5. Track through the dynamic .ply sequence
    tracked_path_3d = []
    
    print("Tracking 3D point across sequence frames...")
    for t in range(num_frames):
        ply_filename = os.path.join(PLY_DIR, f"frame_{t:d}.ply")
        
        # Load absolute 3D coordinates for frame t
        xyz_t = load_xyz_from_ply(ply_filename)
        
        # Extract the positions of our specific anchors at frame t
        anchor_positions = xyz_t[anchor_indices] # Shape [4, 3]
        
        # Calculate the weighted, interpolated 3D coordinate of the tissue
        tracked_pt_t = np.sum(anchor_positions * weights[:, np.newaxis], axis=0)
        tracked_path_3d.append(tracked_pt_t)

    # 6. Save final trajectory
    tracked_path_3d = np.array(tracked_path_3d)
    output_path = "tracked_tissue_path_3d.npy"
    np.save(output_path, tracked_path_3d)
    
    print(f"\nSuccess! Dynamic 3D path extracted directly from .ply files.")
    print(f"Saved trajectory: {output_path} (Shape: {tracked_path_3d.shape})")