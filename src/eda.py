import argparse
from mmdet.datasets import get_dataset
import numpy as np
import mmcv
from mmcv import Config
import os
import cv2
from tqdm import tqdm
import os.path as osp

from src.visualization import draw_bounding_boxes_on_image_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def draw_masks(img, masks):
    if masks is not None:
        for i in range(masks.shape[-1]):
            mask = masks[..., i]
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    return img


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    cfg = Config.fromfile(args.config)
    dataset = get_dataset(cfg.data.train)
    for i in tqdm(np.random.randint(0, len(dataset), 500)):
        data = dataset[i]
        img = data['img'].data.numpy().transpose(1, 2, 0)
        masks = data['gt_masks'].data.transpose(1, 2, 0).astype(bool)
        bboxes = data['gt_bboxes'].data.numpy()
        img = mmcv.imdenormalize(img, mean=cfg.img_norm_cfg.mean, std=cfg.img_norm_cfg.std, to_bgr=False)
        img = draw_masks(img, masks).astype(np.uint8)
        draw_bounding_boxes_on_image_array(img, bboxes, use_normalized_coordinates=False, thickness=5)
        cv2.imwrite(osp.join(args.output, f'{i}_{np.random.randint(0, 10000)}.jpg'), img[..., ::-1])


if __name__ == '__main__':
    main()
