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
import numpy as np
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from torchmetrics.functional.regression import pearson_corrcoef

import lpips
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    lpips_model = lpips.LPIPS(net="vgg", spatial=True).cuda()  # spatial=True: per-pixel map, so
    # lpips_loss can mask out excluded (e.g. tool) regions the same way l1_loss/depth_loss do
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()

    print("opacity_threshold_coarse:", opt.opacity_threshold_coarse)
    print("opacity_threshold_fine_init:", opt.opacity_threshold_fine_init)
    print("opacity_threshold_fine_after:", opt.opacity_threshold_fine_after)
    
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            idx = 0
        else:
            idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images,0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images,0)
        gt_depths = torch.cat(gt_depths, 0)
        masks = torch.cat(masks, 0)
        
        Ll1 = l1_loss(rendered_images, gt_images, masks)
        
        if (gt_depths!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        elif scene.mode == 'binocular':
            rendered_depths[rendered_depths!=0] = 1 / rendered_depths[rendered_depths!=0]
            gt_depths[gt_depths!=0] = 1 / gt_depths[gt_depths!=0]
            depth_loss = l1_loss(rendered_depths, gt_depths, masks)
        elif scene.mode == 'monocular':
            mask_tmp = masks.reshape(-1).bool()
            rendered_flat = rendered_depths.reshape(-1)[mask_tmp]
            gt_flat = gt_depths.reshape(-1)[mask_tmp]
            #scale-shift invariant L1: normalize both to zero mean, unit variance
            r_mean, r_std = rendered_flat.mean(), rendered_flat.std() + 1e-6
            g_mean, g_std = gt_flat.mean(), gt_flat.std() + 1e-6
            rendered_norm = (rendered_flat - r_mean) / r_std
            gt_norm = (gt_flat - g_mean) / g_std
            depth_loss = 0.01 * torch.abs(rendered_norm - gt_norm).mean()
        else:
            raise ValueError(f"{scene.mode} is not implemented.")

        #edge-aware depth smoothness: penalize depth gradients except at image edges
        _d = rendered_depths if rendered_depths.dim() == 4 else rendered_depths.unsqueeze(1)
        #normalize depth for smoothness/TV to prevent loss explosion
        _d_max = _d.max().clamp(min=1e-6)
        _d_norm = _d / _d_max
        img_grad_x = torch.abs(rendered_images[:,:,:,1:] - rendered_images[:,:,:,:-1]).mean(dim=1, keepdim=True)
        img_grad_y = torch.abs(rendered_images[:,:,1:,:] - rendered_images[:,:,:-1,:]).mean(dim=1, keepdim=True)
        #depth smoothness: penalize large depth gradients to prevent radial bias
        depth_grad_x = torch.abs(_d_norm[:,:,:,1:] - _d_norm[:,:,:,:-1])
        depth_grad_y = torch.abs(_d_norm[:,:,1:,:] - _d_norm[:,:,:-1,:])
        depth_smooth_loss = depth_grad_x.mean() + depth_grad_y.mean()

        
        depth_tvloss = TV_loss(_d_norm)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = 0.03 * (img_tvloss + depth_tvloss)

        if (torch.isnan(tv_loss) or torch.isinf(tv_loss)) and not globals().get("_tv_nan_dumped"):
            globals()["_tv_nan_dumped"] = True
            print(f"\n[ITER {iteration}] TV-NAN DIAGNOSTIC (one-time dump):")
            print(f"  img_tvloss={img_tvloss.item()}, depth_tvloss={depth_tvloss.item()}")
            print(f"  _d_max={_d_max.item()}, rendered_depths: isnan={torch.isnan(rendered_depths).any().item()}, "
                  f"isinf={torch.isinf(rendered_depths).any().item()}, min={rendered_depths.min().item():.4g}, max={rendered_depths.max().item():.4g}")
            mask_bool = masks.bool() if masks.dtype != torch.bool else masks
            unmasked_d = rendered_depths[~mask_bool] if mask_bool.shape == rendered_depths.shape else None
            if unmasked_d is not None and unmasked_d.numel() > 0:
                print(f"  rendered_depths WITHIN masked-out region: isnan={torch.isnan(unmasked_d).any().item()}, "
                      f"isinf={torch.isinf(unmasked_d).any().item()}, min={unmasked_d.min().item():.4g}, max={unmasked_d.max().item():.4g}")
            masked_in_d = rendered_depths[mask_bool] if mask_bool.shape == rendered_depths.shape else None
            if masked_in_d is not None and masked_in_d.numel() > 0:
                print(f"  rendered_depths WITHIN masked-in region: isnan={torch.isnan(masked_in_d).any().item()}, "
                      f"isinf={torch.isinf(masked_in_d).any().item()}, min={masked_in_d.min().item():.4g}, max={masked_in_d.max().item():.4g}")
            print(f"  rendered_images: isnan={torch.isnan(rendered_images).any().item()}, isinf={torch.isinf(rendered_images).any().item()}")
        
        loss = Ll1 + depth_loss + tv_loss + 0.05

        psnr_ = psnr(rendered_images, gt_images, masks).mean().double()        
        
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(2e-2, 2e-2, 2e-2)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(rendered_images,gt_images)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if opt.lambda_lpips !=0:
            lpipsloss = lpips_loss(rendered_images,gt_images,lpips_model,masks)
            loss += opt.lambda_lpips * lpipsloss
        
        if gaussians.get_xyz.shape[0] == 0:
            print(f"Warning: Step {iteration} has 0 active Gaussians. Skipping backward pass to prevent CUDA overflow.")
            gaussians.optimizer.zero_grad(set_to_none=True)
        elif torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
            print(f"[ITER {iteration}] Bad loss ({loss.item():.2e})! "
                    f"Ll1={Ll1.item():.2e}, depth={depth_loss.item():.2e}, "
                    f"tv={tv_loss.item():.2e}, smooth={depth_smooth_loss.item():.2e}. Skipping.")
            gaussians.optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            # NOTE: previously only clipped param_groups[0] (xyz). The deformation/grid
            # parameter groups (param_groups[1:3], both LR-scaled by spatial_lr_scale ==
            # cameras_extent) had no gradient-norm protection at all -- on a scene with a
            # large real-world coordinate scale (ours), this let the HexPlane grid
            # parameters explode to NaN within a few hundred fine-stage iterations, with
            # compute_regulation() only surfacing the pre-existing NaN, not causing it.
            for group in gaussians.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            if viewspace_point_tensor_list[idx].grad is not None:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            else:
                print(f"Warning: Step {iteration} has no gradient for viewspace_point_tensor. This may indicate that all points are inactive or pruned. Skipping gradient accumulation for this frame.")
                pass
        iter_end.record()

        with torch.no_grad():
            if stage == "fine" and iteration % 250 == 0:
                op = gaussians.get_opacity
                print(f"\n[ITER {iteration}] opacity stats: min={op.min().item():.4f} "
                      f"median={op.median().item():.4f} mean={op.mean().item():.4f} max={op.max().item():.4f} "
                      f"frac>0.1={ (op > 0.1).float().mean().item():.3f} frac>0.5={ (op > 0.5).float().mean().item():.3f}")
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            #densification
            if iteration < opt.densify_until_iter :
                #keep track of max radii in image-space for pruning
                try:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                except RuntimeError:
                    print(f"\n[ITER {iteration}] DIAGNOSTIC DUMP before re-raising:")
                    print(f"  gaussians._xyz: shape={gaussians._xyz.shape}, "
                          f"isnan={torch.isnan(gaussians._xyz).any().item()}, isinf={torch.isinf(gaussians._xyz).any().item()}, "
                          f"min={gaussians._xyz.min().item():.4g}, max={gaussians._xyz.max().item():.4g}")
                    scaling = gaussians.get_scaling
                    print(f"  gaussians.get_scaling: isnan={torch.isnan(scaling).any().item()}, isinf={torch.isinf(scaling).any().item()}, "
                          f"min={scaling.min().item():.4g}, max={scaling.max().item():.4g}")
                    print(f"  radii: shape={radii.shape}, dtype={radii.dtype}, "
                          f"min={radii.min().item()}, max={radii.max().item()}")
                    print(f"  visibility_filter: shape={visibility_filter.shape}, sum={visibility_filter.sum().item()}")
                    print(f"  max_radii2D: shape={gaussians.max_radii2D.shape}")
                    raise
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if gaussians._xyz.shape[0] < opt.max_gaussians:
                        gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, max_gaussians=opt.max_gaussians)
                    else:
                        print(f"[ITER {iteration}] Point cap reached ({gaussians._xyz.shape[0]}), skipping densify")
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    if not args.no_fine:
        # Scoped to fine stage only (not coarse, which is already confirmed healthy across
        # 3000 iterations) -- detect_anomaly adds significant per-backward overhead, and
        # coarse doesn't need it. This makes the first backward() that produces a NaN/Inf
        # gradient raise immediately with a full stack trace pointing at the exact op,
        # instead of silently propagating NaN through _xyz/_scaling as it has been.
        # Expect this run to CRASH with a traceback within the first ~200 fine iterations
        # (that's the diagnostic signal, not a new bug) -- the traceback's deepest frame
        # inside this repo's own code (not torch internals) is what to report back.
        torch.autograd.set_detect_anomaly(True)
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    '''
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    mask = viewpoint.mask.to("cuda")
                    
                    image, gt_image, mask = image.unsqueeze(0), gt_image.unsqueeze(0), mask.unsqueeze(0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
        '''

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3000,4000, 5000, 6000, 9000, 10000, 14000, 20000, 30_000,45000,60000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")
