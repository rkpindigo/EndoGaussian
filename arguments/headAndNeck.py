ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10
)

OptimizationParams = dict(
    coarse_iterations = 3000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 6000,
    percent_dense = 0.002,
    
    opacity_reset_interval = 999999, #turning off opacity reset for now

    densify_from_iter = 500,
    densify_until_iter = 4000, #caps point-generation well into fine stage
    densify_grad_threshold = 0.0002, #only densify high-confidence geometric edges

    position_lr_max_steps = 6000,
    prune_interval = 100, #aggresive pruning to clean up floaters instead of relying on opacity thresholding
    opacity_threshold_coarse=0.001, #filter out faint floaters aggressively
    opacity_threshold_fine_init = 0.001,
    opacity_threshold_fine_after = 0.001
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 64,
     'resolution': [64, 64, 64, 100]
    },
    grid_bounds = [[-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
)