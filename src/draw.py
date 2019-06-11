import argparse
import numpy as np
from multiprocessing import Pool
import cv2
import os
import json
from tqdm import tqdm
import mmcv
import os.path as osp
from functools import partial
import pycocotools.mask as mutils
import pandas as pd

N_CLASSES = 46


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--predictions', type=str)
    parser.add_argument('--submission', type=str, default=None)
    parser.add_argument('--classes', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_jobs', type=int, default=80)
    parser.add_argument('--metric_threshold', type=float, default=0.1)
    return parser.parse_args()


def get_spaced_colors(n, start_color=(75, 0, 130)):
    r, g, b = start_color
    step = 256 / n
    colors = []
    for i in range(n):
        r += step
        g += step
        b += step
        colors.append((int(r) % 256, int(g) % 256, int(b) % 256))
    return np.random.permutation(colors).reshape(-1, 1, 3).astype(np.uint8)


def put_text(img, color, text, i, x_shift=10, y_shift=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = x_shift, y_shift + (i + 1) * text_size[1] * 2
    img = cv2.putText(img, text, (text_x, text_y), font, 1.5, tuple(map(int, color)), 3)
    return img


def draw_masks(img, masks, colors, classes):
    if masks is not None:
        assert len(colors) == len(masks)
        mask_colors = [colors[i] for i, mask in enumerate(masks) if mask]
        mask_classes = [classes[i] for i, mask in enumerate(masks) if mask]
        masks = mmcv.concat_list(masks)
        for i, (mask, color, cls) in enumerate(zip(masks, mask_colors, mask_classes)):
            mask = mutils.decode(mask).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color * 0.5
            img = put_text(img, color[0], cls, i)
    return img


def get_gt_masks(annotation):
    gt_masks = [[] for _ in range(N_CLASSES)]
    for mask, label in zip(annotation['ann']['masks'], annotation['ann']['labels']):
        gt_masks[label - 1].append(mask)
    return gt_masks


def draw(args, root, output, metric_threshold, colors, classes):
    prediction, annotation, metric = args
    if metric is None:
        output_filename = annotation['filename']
    elif metric < metric_threshold:
        output_filename = f"{100 * metric:0.3f}_{annotation['filename']}"
    else:
        return None

    img = cv2.imread(osp.join(root, annotation['filename']))[..., ::-1]
    _, prediction_masks = prediction
    prediction_img = draw_masks(img.copy(), prediction_masks, colors, classes)
    gt_img = draw_masks(img.copy(), get_gt_masks(annotation), colors, classes)

    output_image = np.hstack([img, gt_img, prediction_img])
    return cv2.imwrite(osp.join(output, output_filename), output_image[..., ::-1])


def main():
    args = parse_args()
    predictions = mmcv.load(args.predictions)
    annotation = mmcv.load(args.annotation)
    classes = json.load(open(args.classes))
    classes = [x['name'] for x in classes['categories']]

    if args.submission is not None:
        submission = pd.read_csv(args.submission)
        submission = submission.drop_duplicates('ImageId')
        id2metric = dict(zip(submission['ImageId'], submission['mAP']))
        metrics = [id2metric[x['filename']] for x in annotation]
    else:
        metrics = [None for _ in annotation]

    os.makedirs(args.output, exist_ok=True)
    colors = get_spaced_colors(N_CLASSES)
    partial_draw = partial(
        draw,
        root=args.root,
        output=args.output,
        metric_threshold=args.metric_threshold,
        colors=colors,
        classes=classes
    )
    with Pool(args.n_jobs) as p:
        list(
            tqdm(
                iterable=p.imap_unordered(partial_draw, zip(predictions, annotation, metrics)), total=len(predictions)
            )
        )


if __name__ == '__main__':
    main()
