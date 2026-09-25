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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine"):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    
    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point],
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point])

        if not globals().get("_deform_nan_dumped"):
            bad_out = any(torch.isnan(t).any() or torch.isinf(t).any() for t in
                          [means3D_deform, scales_deform, rotations_deform, opacity_deform])
            if bad_out:
                globals()["_deform_nan_dumped"] = True
                print("\nDEFORM-NAN DIAGNOSTIC (one-time dump):")
                for name, t in [("means3D_in", means3D[deformation_point]), ("scales_in", scales[deformation_point]),
                                ("rotations_in", rotations[deformation_point]), ("opacity_in", opacity[deformation_point]),
                                ("time_in", time[deformation_point])]:
                    print(f"  {name}: isnan={torch.isnan(t).any().item()}, isinf={torch.isinf(t).any().item()}, "
                          f"min={t.min().item():.4g}, max={t.max().item():.4g}")
                for name, t in [("means3D_deform", means3D_deform), ("scales_deform", scales_deform),
                                ("rotations_deform", rotations_deform), ("opacity_deform", opacity_deform)]:
                    isnan = torch.isnan(t).any().item()
                    isinf = torch.isinf(t).any().item()
                    finite = t[torch.isfinite(t)]
                    print(f"  {name}: isnan={isnan}, isinf={isinf}, "
                          f"finite_min={finite.min().item() if finite.numel() else 'n/a'}, "
                          f"finite_max={finite.max().item() if finite.numel() else 'n/a'}")
                grids = pc._deformation.deformation_net.grid.grids
                for gi, g in enumerate(grids):
                    for pi, plane in enumerate(g):
                        if torch.isnan(plane).any() or torch.isinf(plane).any():
                            print(f"  grid[{gi}][{pi}]: CONTAINS nan/inf, shape={plane.shape}")
                        else:
                            print(f"  grid[{gi}][{pi}]: clean, min={plane.min().item():.4g}, max={plane.max().item():.4g}")

        # Clamp deformation outputs to prevent illegal memory access in rasterizer.
        # This scene's real point-cloud extent is roughly x:[-120,61], y:[-49,158],
        # z:[-256,-31] (see scene/__init__.py's set_aabb call) -- both bounds below are
        # sized relative to that, not arbitrary round numbers.
        #
        # means3D_deform: tightened from +/-1e3 (4x the scene's own extent) to +/-500
        # (~2x), still a generous safety margin beyond any real deformation.
        means3D_deform = torch.clamp(means3D_deform, -500.0, 500.0)
        # scales_deform is in log-space (scaling_activation = torch.exp in
        # GaussianModel). The original (-20,20) bound let the actual exponentiated scale
        # range from ~2e-9 to ~4.85e8; either extreme singularizes a Gaussian's
        # projected 2D covariance matrix, which is what caused
        # "_RasterizeGaussiansBackward returned nan values in its 0th output"
        # (grad_means3D). A first tightening to (-10,10) (actual range ~4.5e-5 to
        # ~22026 -- still ~90x the scene's own extent) reduced but did not eliminate
        # the crash; it recurred at a later iteration under the same signature.
        # Tightened further to (-5,5): actual scale range ~0.0067 to ~148, comfortably
        # covering any real Gaussian in this scene without allowing anything close to
        # scene-covering degenerate blobs.
        scales_deform = torch.clamp(scales_deform, -5.0, 5.0)
        rotations_deform = torch.clamp(rotations_deform, -2.0, 2.0)
        opacity_deform = torch.clamp(opacity_deform, -20.0, 20.0)
    # print(time.max())
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    # BUG FIX: this used to read `pc.opacity_activation(opacity)`, re-using the raw
    # STATIC opacity from line 60 (opacity = pc._opacity) instead of opacity_final (the
    # deformation-blended tensor built just above from opacity_deform +
    # opacity[~deformation_point]). That silently discarded the deformation network's
    # per-timestep opacity output entirely -- the rasterizer only ever received the
    # static, time-invariant opacity, for every frame, in every render call.
    opacity = pc.opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,}

