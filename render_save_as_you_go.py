video_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    ours_writer = imageio.get_writer(os.path.join(video_dir, 'ours_video.mp4'), fps=30, quality=8)
    gt_writer = imageio.get_writer(os.path.join(video_dir, 'gt_video.mp4'), fps=30, quality=8)

    reconstruct_data = [] if reconstruct else None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        stage = 'coarse' if no_fine else 'fine'
        rendering = render(view, gaussians, pipeline, background, stage=stage)
        rendered_image = rendering["render"].cpu()
        rendered_depth = rendering["depth"].cpu()

        torchvision.utils.save_image(rendered_image, os.path.join(render_path), '{0:05d}.png'.format(idx))
        depth_np = np.clip(rendered_depth.squeeze().numpy().astype(np.uint8), 0, 255)
        cv2.imwrite(os.path.join(depth_path, '{0:05d}.png'.format(idx)), depth_np)

        frame = (rendered_image.permute(1,2,0).numpy() * 255).clip(0, 255).astype(np.uint8)
        ours_writer.append_data(frame)