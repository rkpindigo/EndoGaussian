#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams

class Scene:

    gaussians : GaussianModel
    
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.mode = args.mode

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        if os.path.exists(os.path.join(args.source_path, "sparse")) and args.extra_mark is None:
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'endonerf':
            scene_info = sceneLoadTypeCallbacks["endonerf"](args.source_path, mode=args.mode, downsample=getattr(args, "endo_downsample", 1.0))
            print("Found poses_bounds.py and extra marks with EndoNeRf")
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'hamlyn':
            scene_info = sceneLoadTypeCallbacks["hamlyn"](args.source_path, mode=args.mode)
            print("Found poses_bounds.py and extra marks with EndoNeRf")
        elif os.path.exists(os.path.join(args.source_path, "point_cloud.obj")) or os.path.exists(os.path.join(args.source_path, "left_point_cloud.obj")):
            scene_info = sceneLoadTypeCallbacks["scared"](args.source_path, mode=args.mode, init_pts=args.init_pts)
            print("Found point_cloud.obj, assuming SCARED data!")
        else:
            assert False, "Could not recognize scene type!"
        
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # Point-cloud-derived extent, used both as a degenerate-camera-motion fallback
        # and as a floor on the camera-motion-derived value (see below). Use a
        # percentile of per-point distance-from-centroid rather than a per-axis
        # min/max (or per-axis percentile) bounding box. The initial point cloud comes
        # from unprojecting depth through a 1/x disparity inversion (see
        # scene/endo_loader.py) combined with a MULTIPLICATIVE per-pixel factor
        # (X = (u-cx)/fx * depth, Y = (v-cy)/fy * depth) -- a pixel near the frame edge
        # combined with only a moderately elevated depth (not an outlier on its own)
        # can still land far from the scene center once position and depth compound.
        # Per-axis clipping doesn't catch that, since such a point can be within-range
        # on each axis independently while still being unusually far away jointly. A
        # single combined per-point radius avoids that.
        pts = scene_info.point_cloud.points
        centroid = np.median(pts, axis=0)
        radii = np.linalg.norm(pts - centroid, axis=1)
        pointcloud_extent = float(np.percentile(radii, 90))

        if self.cameras_extent < 1e-6:
            # Degenerate under static/identity poses -- every camera center is
            # identical, so the camera-spread-based radius (diagonal * 1.1 in
            # getNerfppNorm) is exactly zero. spatial_lr_scale (set from this
            # value in create_from_pcd) directly multiplies the position/
            # deformation/grid learning rates in training_setup(), so a zero
            # here silently freezes all of them for the entire training run --
            # this is what caused the flat-plane collapse on static-pose scenes.
            print(f"self.cameras_extent computed as {self.cameras_extent} (degenerate, likely static poses) "
                  f"-- falling back to point-cloud-derived extent {pointcloud_extent}")
            self.cameras_extent = pointcloud_extent
        elif self.cameras_extent < pointcloud_extent:
            # Non-degenerate but still too small: cameras_extent only reflects how far
            # the CAMERA physically moved, which is unrelated to how large the visible
            # GEOMETRY is. For a scene where the camera translates very little while
            # looking across a large depth range (e.g. a near-static endoscope), camera
            # motion can be a small fraction of the true scene scale. This matters
            # because cameras_extent is used as a stand-in for scene scale in more than
            # just the LR scaling: GaussianModel.prune() (scene/gaussian_model.py)
            # removes any Gaussian whose scale exceeds 0.1*extent once
            # iteration > opacity_reset_interval -- with too-small extent, that
            # threshold is far tighter than the geometry actually needs, causing mass
            # pruning right at the reset boundary (confirmed: 11698 -> 621 Gaussians
            # between iterations 3000 and 4000 on this scene). Floor cameras_extent at
            # the point-cloud extent so it never under-represents the real scene size.
            print(f"self.cameras_extent computed as {self.cameras_extent} from camera motion, which is smaller "
                  f"than the point-cloud extent {pointcloud_extent} -- flooring to the latter so scale-dependent "
                  f"thresholds (prune size, spatial_lr_scale) reflect the true scene size, not just camera travel.")
            self.cameras_extent = pointcloud_extent
        print("self.cameras_extent is ", self.cameras_extent)

        print("Loading Training Cameras")
        self.train_camera = scene_info.train_cameras
        print("Loading Test Cameras")
        self.test_camera = scene_info.test_cameras
        print("Loading Video Cameras")
        self.video_camera = scene_info.video_cameras
        
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)

        if self.loaded_iter:
            iteration_str = 'iteration_'+str(self.loaded_iter) if not load_coarse else 'coarse_iteration_'+str(self.loaded_iter)
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           iteration_str,
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    iteration_str,
                                                   ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera