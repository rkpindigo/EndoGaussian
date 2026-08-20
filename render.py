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
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import open3d as o3d
from utils.graphics_utils import fov2focal
import cv2

import os

cpu_list = [0]
os.sched_setaffinity(0, cpu_list)



to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def export_gaussian_tissue_point_cloud(gaussians, views, model_path, name, iteration):
    """
    Exports 3D Gaussian positions for each frame directly at its time step, 
    preserving the tissue hidden behind tools and eliminating 2D edge artifacts.
    """
    output_folder = os.path.join(model_path, name, f"ours_{iteration}", "full_tissue_ply")
    os.makedirs(output_folder, exist_ok=True)
    
    print("Exporting full 3D Gaussian tissue point clouds...")
    for idx, view in enumerate(tqdm(views, desc="Exporting Point Clouds")):
        means3D = gaussians.get_xyz

        if hasattr(gaussians, '_deformation') and hasattr(gaussians, '_deformation_table'):
            time = torch.tensor(view.time).to(means3D.device).repeat(means3D.shape[0], 1)
            scales = gaussians._scaling
            rotations = gaussians._rotation
            opacity = gaussians._opacity
            deformation_point = gaussians._deformation_table

            means3D_deform, sclaes_deform, rotations_deform, opacity_deform = gaussians._deformation(
                means3D[deformation_point],
                scales[deformation_point],
                rotations[deformation_point],
                opacity[deformation_point],
                time[deformation_point]
            )

            means3D_final = torch.zeros_like(means3D)
            means3D_final[deformation_point] = means3D_deform
            means3D_final[~deformation_point] = means3D[~deformation_point]
        
        else:
            means3D_final = means3D

        
        xyz = means3D_final.detach().cpu().numpy()

        shs = gaussians.get_features[:,0,:].detach().cpu().numpy()
        colors = np.clip(shs * 0.28209479177387814 + 0.5, 0, 1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        save_path = os.path.join(output_folder, f"frame_{idx:05d}.ply")
        o3d.io.write_point_cloud(save_path, pcd)
    
    print("Done with reconstructions")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, no_fine, render_test=True, reconstruct=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    
    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        stage = 'coarse' if no_fine else 'fine'
        rendering = render(view, gaussians, pipeline, background, stage=stage)
        for i in range(len(render_depths)):
            print(i, render_depths[i].min().item(), render_depths[i].max().item(), render_depths[i].mean().item())
        render_depths.append(rendering["depth"].cpu())
        render_images.append(rendering["render"].cpu())
        if name in ["train", "test", "video"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)
            mask = view.mask
            mask_list.append(mask)
            gt_depth = view.original_depth
            gt_depths.append(gt_depth)
    
    if render_test:
        test_times = 50
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx == 0 and i == 0:
                    time1 = time()
                stage = 'coarse' if no_fine else 'fine'
                rendering = render(view, gaussians, pipeline, background, stage=stage)
        time2=time()
        print("FPS:",(len(views)-1)*test_times/(time2-time1))
    
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image = np.clip(image.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image = image.cpu().squeeze().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(gtdepth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    
    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    
    if reconstruct:
        # Render with white background to recover per-pixel alpha
        white_bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        white_images = []
        for idx, view in enumerate(tqdm(views, desc="Rendering white bg for alpha")):
            stage = 'coarse' if no_fine else 'fine'
            white_render = render(view, gaussians, pipeline, white_bg, stage=stage)
            white_images.append(white_render["render"].cpu())
        world_view_transforms = [view.world_view_transform.cpu().numpy() for view in views]
        #keep masks=mask_list for enabling masks, change to masks=None for disabling masks
        reconstruct_point_cloud(render_images, white_images, render_depths, camera_parameters, name, masks=mask_list, world_view_transforms=world_view_transforms)
        #reconstruct_point_cloud(render_images, white_images, render_depths, camera_parameters, name, masks=None, world_view_transforms=world_view_transforms)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, reconstruct: bool):
    
    cpu_list = list(range(cpu_count))[1:2]
    psutil.Process().cpu_affinity(cpu_list)
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_coarse=dataset.no_fine)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.no_fine, reconstruct=reconstruct)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.no_fine, render_test=True, reconstruct=reconstruct)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, dataset.no_fine, render_test=True, reconstruct=reconstruct)

def _unproject_to_world(depth, valid_mask, wvt, focal_x, focal_y, width, height):
    """depth: (H,W). valid_mask: (H,W) bool. wvt: (4,4) world_view_transform (world->camera,
    row-vector convention, i.e. p_cam_row = p_world_row_homog @ wvt). Returns (N,3) world-space
    points and (N,3)-index-aligned pixel (u,v) arrays for the True entries of valid_mask."""
    vs, us = np.nonzero(valid_mask)
    z = depth[vs, us]
    cx, cy = width / 2.0, height / 2.0
    x_cam = (us - cx) * z / focal_x
    y_cam = (vs - cy) * z / focal_y
    cam_pts_h = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
    world_pts_h = cam_pts_h @ np.linalg.inv(wvt)
    return world_pts_h[:, :3], us, vs


def _reproject_from_world(world_pts, wvt, focal_x, focal_y, width, height):
    """world_pts: (N,3). wvt: (4,4) TARGET frame's world_view_transform. Returns float (u,v)
    pixel coords, camera-space depth, and a validity mask (points in front of the camera)."""
    n = world_pts.shape[0]
    world_pts_h = np.concatenate([world_pts, np.ones((n, 1))], axis=1)
    cam_pts_h = world_pts_h @ wvt
    x_cam, y_cam, z_cam = cam_pts_h[:, 0], cam_pts_h[:, 1], cam_pts_h[:, 2]
    cx, cy = width / 2.0, height / 2.0
    valid = z_cam > 1e-8
    u = np.where(valid, focal_x * x_cam / np.where(valid, z_cam, 1) + cx, -1.0)
    v = np.where(valid, focal_y * y_cam / np.where(valid, z_cam, 1) + cy, -1.0)
    return u, v, z_cam, valid


def _bilateral_clip(depth_np):
    """Same bilateral-filter + percentile-clip cleanup the original per-frame pipeline always
    did -- given a depth map with holes (if any) already zeroed. Runs on ONE frame's own depth
    only, so it never gets influenced by what's happening in a different frame's fill-in."""
    import cv2
    zero_mask = (depth_np == 0)
    depth_smoother = (128, 64, 64)
    out = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
    valid_depth = out[out != 0]
    if valid_depth.size == 0:
        return out, False
    close_depth = np.percentile(valid_depth, 5)
    inf_depth = np.percentile(valid_depth, 95)
    out = np.clip(out, close_depth, inf_depth)
    out[zero_mask] = 0
    return out, True


def reconstruct_point_cloud(black_images, white_images, depths, camera_parameters, name, alpha_threshold=0.5, masks=None, world_view_transforms=None, temporal_fill_radius=30):
    import copy
    output_frame_folder = os.path.join("reconstruct", name)
    os.makedirs(output_frame_folder, exist_ok=True)
    frames = np.arange(len(black_images))
    focal_x, focal_y, width, height = camera_parameters
    n_frames = len(frames)

    # Pass 1: for each frame, compute two FULLY cleaned (alpha/uniform-dark zeroed, bilateral
    # filtered, percentile clipped) depth maps -- one with the tool mask applied (matching the
    # original per-frame pipeline exactly, used as both the default output and the source of
    # truth for reprojection into other frames) and one WITHOUT the mask applied (the model's
    # own native prediction, used only as a last-resort fallback in pass 2 for pixels no
    # neighboring frame can fill). Doing the filter/clip here, per-frame, in isolation is what
    # keeps a fill in one frame's holes from ever changing another region's global statistics.
    cleaned_rgbs = []
    masked_depths = []      # holes zeroed, then bilateral+clip -- old "simple exclusion" output
    unmasked_depths = []    # never mask-zeroed, then bilateral+clip -- fallback source
    mask_arrays = []
    world_points = []       # world_points[i] = (N_i, 3) array (from masked_depths), or None
    world_colors = []
    for i_frame in frames:
        black_np = black_images[i_frame].permute(1, 2, 0).numpy()
        white_np = white_images[i_frame].permute(1, 2, 0).numpy()
        final_T = (white_np - black_np).mean(axis=2)
        alpha_map = 1.0 - np.clip(final_T, 0, 1)

        rgb_tensor = black_images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()
        depth_np = depth_np.squeeze(0)

        depth_np[alpha_map < alpha_threshold] = 0

        rgb_float = rgb_np.astype(np.float32)
        brightness = rgb_float.mean(axis=2)
        max_ch = rgb_float.max(axis=2)
        min_ch = rgb_float.min(axis=2)
        channel_diff = max_ch - min_ch
        is_uniform_dark = (brightness < 25) & (channel_diff < 12)
        depth_np[is_uniform_dark] = 0

        cleaned_rgbs.append(rgb_np)

        if masks is not None:
            mask_np = masks[i_frame]
            if hasattr(mask_np, "cpu"):
                mask_np = mask_np.cpu().numpy()
            mask_np = np.squeeze(mask_np)
        else:
            mask_np = None
        mask_arrays.append(mask_np)

        depth_masked = depth_np.copy()
        if mask_np is not None:
            depth_masked[mask_np == 0] = 0
        depth_masked, _ = _bilateral_clip(depth_masked)
        masked_depths.append(depth_masked)

        depth_unmasked, _ = _bilateral_clip(depth_np.copy())
        unmasked_depths.append(depth_unmasked)

        if world_view_transforms is not None and mask_np is not None:
            valid = (mask_np != 0) & (depth_masked != 0)
            wpts, us, vs = _unproject_to_world(depth_masked, valid, world_view_transforms[i_frame],
                                                focal_x, focal_y, width, height)
            world_points.append(wpts)
            world_colors.append(rgb_np[vs, us])
        else:
            world_points.append(None)
            world_colors.append(None)

    for i_frame in frames:
        depth_np = masked_depths[i_frame].copy()
        rgb_new = cleaned_rgbs[i_frame].copy()

        # Fill tool-masked holes by reprojecting nearby-in-time frames' actual observed
        # geometry through this frame's own camera pose -- correct for moving cameras, not
        # just static ones (a static camera is just the degenerate case where every frame's
        # world_view_transform is identical, so this reduces to a same-pixel lookup then).
        # Doesn't correct for tissue deformation between source and target frame, so accuracy
        # degrades the further out in time a fill has to search.
        mask_np = mask_arrays[i_frame]
        if mask_np is not None:
            hole_mask = (mask_np == 0)
            if hole_mask.any() and world_view_transforms is not None:
                still_missing = hole_mask.copy()
                target_wvt = world_view_transforms[i_frame]
                for offset in range(1, temporal_fill_radius + 1):
                    if not still_missing.any():
                        break
                    for neighbor in (i_frame - offset, i_frame + offset):
                        if neighbor < 0 or neighbor >= n_frames or not still_missing.any():
                            continue
                        wpts = world_points[neighbor]
                        if wpts is None or wpts.shape[0] == 0:
                            continue
                        u, v, z_cam, reproj_valid = _reproject_from_world(
                            wpts, target_wvt, focal_x, focal_y, width, height)
                        u_int = np.round(u).astype(int)
                        v_int = np.round(v).astype(int)
                        in_bounds = reproj_valid & (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
                        if not in_bounds.any():
                            continue
                        cu, cv, cz = u_int[in_bounds], v_int[in_bounds], z_cam[in_bounds]
                        crgb = world_colors[neighbor][in_bounds]
                        keep = still_missing[cv, cu]
                        if not keep.any():
                            continue
                        cu, cv, cz, crgb = cu[keep], cv[keep], cz[keep], crgb[keep]
                        # z-buffer: process farthest-first so the nearest candidate is written
                        # last and wins wherever multiple candidates land on the same pixel
                        order = np.argsort(-cz)
                        cu, cv, cz, crgb = cu[order], cv[order], cz[order], crgb[order]
                        depth_np[cv, cu] = cz
                        rgb_new[cv, cu] = crgb
                        still_missing[cv, cu] = False
                if still_missing.any():
                    # no neighbor within the search radius had this region unmasked (e.g. a
                    # tool that sat in the same spot for the whole window) -- fall back to the
                    # model's own native prediction there rather than leaving a hole
                    fallback_depth = unmasked_depths[i_frame]
                    fallback_rgb = cleaned_rgbs[i_frame]
                    has_fallback = still_missing & (fallback_depth != 0)
                    depth_np[has_fallback] = fallback_depth[has_fallback]
                    rgb_new[has_fallback] = fallback_rgb[has_fallback]
                    depth_np[still_missing & ~has_fallback] = 0
            elif hole_mask.any():
                # no pose info available -- fall back to the model's own native prediction
                fallback_depth = unmasked_depths[i_frame]
                has_fallback = hole_mask & (fallback_depth != 0)
                depth_np[has_fallback] = fallback_depth[has_fallback]
                rgb_new[has_fallback] = cleaned_rgbs[i_frame][has_fallback]
                depth_np[hole_mask & ~has_fallback] = 0

        zero_mask = (depth_np == 0)

        if (~zero_mask).sum() == 0:
            print(f"Warning: Frame {i_frame} has completely empty depth map. Skipping.")
            continue

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, height / 2),
            project_valid_depth_only=True
        )

        o3d.io.write_point_cloud(os.path.join(output_frame_folder, 'frame_{}.ply'.format(i_frame)), pcd)

if __name__ == "__main__":
    import psutil

    cpu_count = psutil.cpu_count()
    print(cpu_count)

    cpu_list = list(range(cpu_count))[1:2]

    psutil.Process().cpu_affinity(cpu_list)

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--reconstruct", action="store_true")
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.reconstruct)