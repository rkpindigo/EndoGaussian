import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree
import os

def load_xyz_from_ply(ply_path):
    """Extracts just the (x, y, z) coordinates from a 3DGS .ply file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    # Extract x, y, z fields as a numpy array
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    return xyz

# 1. Load the first frame (canonical/initial state)
ply_t0_path = "/media/storage1/rpatro/repos/EndoGaussian/reconstruct/video/frame_0.ply"
xyz_t0 = load_xyz_from_ply(ply_t0_path)

# 2. unprojected 3D laser coordinate
# P_world_0 = unproject_pixel_to_3d(...)
P_world_0 = np.array([X, Y, Z]) 

# 3. Build tree and find the anchor indices
tree = KDTree(xyz_t0)
distances, anchor_indices = tree.query(P_world_0, k=4)

# 4. Calculate tracking weights
weights = 1.0 / (distances + 1e-6)
weights /= weights.sum()

print(f"Anchor Indices locked in: {anchor_indices}")
print(f"Interpolation Weights: {weights}")

############################################################################################################### 

# Directory containing exported reconstructed .ply files
ply_dir = "/media/storage1/rpatro/repos/EndoGaussian/reconstruct/video"
num_frames = 100 # Adjust to actual sequence length

tracked_path_3d = []

for t in range(num_frames):
    ply_filename = os.path.join(ply_dir, f"frame_{t:04d}.ply") # Adjust naming format if needed
    
    # Load coordinates for this specific frame
    xyz_t = load_xyz_from_ply(ply_filename)
    
    # Extract the positions of our specific anchors at frame t
    anchor_positions = xyz_t[anchor_indices] # Shape [4, 3]
    
    # Calculate the weighted 3D coordinate of the marked tissue
    tracked_pt_t = np.sum(anchor_positions * weights[:, np.newaxis], axis=0)
    
    tracked_path_3d.append(tracked_pt_t)

tracked_path_3d = np.array(tracked_path_3d) # Shape [T, 3]
np.save("tracked_tissue_path_3d.npy", tracked_path_3d)
print("Finished! Dynamic 3D path extracted directly from .ply files.")