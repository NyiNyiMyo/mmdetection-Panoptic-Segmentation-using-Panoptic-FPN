@torch.no_grad()
def visualize_mmdet_panoptic_fpn(
    model,
    image_paths,
    num_images=3,
):
    import matplotlib.pyplot as plt
    import random
    import os
    import mmcv
    from mmdet.apis import inference_detector
    from mmdet.registry import VISUALIZERS

    model.eval()

    # -------------------------
    # IMAGE LIST
    # -------------------------
    all_imgs = [os.path.join(image_paths, f)
                for f in os.listdir(image_paths)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    sample_imgs = random.sample(all_imgs, num_images)

    plt.figure(figsize=(6 * num_images, 6))

    # -------------------------
    # BUILD VISUALIZER ONCE
    # -------------------------
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    for idx, img_path in enumerate(sample_imgs):

        # -------------------------
        # INFERENCE
        # -------------------------
        result = inference_detector(model, img_path)

        # -------------------------
        # LOAD IMAGE
        # -------------------------
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        # 🔥 IMPORTANT: match resolution
        pan_map = result.pred_panoptic_seg.sem_seg
        H, W = pan_map.shape[-2:]
        img = mmcv.imresize(img, (W, H))

        # -------------------------
        # DRAW (demo pipeline)
        # -------------------------
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            show=False
        )

        vis = visualizer.get_image()

        # -------------------------
        # SHOW
        # -------------------------
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(vis)
        plt.axis("off")

    plt.tight_layout()
    plt.show()