ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10,
    mode = 'monocular',
    endo_downsample=2.0
)

OptimizationParams = dict(
    coarse_iterations = 3000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 10000,
    percent_dense = 0.01,

    lambda_dssim=0.2,
    
    opacity_reset_interval = 3000,

    densify_from_iter = 500,
    densify_until_iter = 2500,
    densify_grad_threshold_coarse = 0.005,
    densify_grad_threshold_fine_init = 0.005,
    densify_grad_threshold_after = 0.005,

    position_lr_max_steps = 9000,
    prune_interval = 100,
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 100]
    },
    grid_bounds = [[-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
    multires = [1, 2, 4],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.005,
    l1_time_planes =  0.0001,
    weight_decay_iteration=0,
)