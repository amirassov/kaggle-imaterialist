import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
import cv2
from tqdm import tqdm
import mmcv
import pycocotools.mask as mutils
from src.rle import kaggle_rle_encode
from src.metric import calc_score_per_class
from src.utils import create_labeled_mask, check_overlaps, hard_overlaps_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--predictions', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_jobs', type=int, default=80)
    parser.add_argument('--add_metric', action='store_true')
    return parser.parse_args()


def decode_and_resize(
    masks, x_min=None, x_max=None, y_min=None, y_max=None, original_height=None, original_width=None
):
    binary_mask = mutils.decode(masks)
    if x_min is not None:
        crop_height, crop_width, channels = binary_mask.shape
        assert crop_height == y_max - y_min
        assert crop_width == x_max - x_min
        original_mask = np.zeros((original_height, original_width, channels))
        original_mask[y_min:y_max, x_min:x_max] = binary_mask
        binary_mask = original_mask
    binary_mask = cv2.resize(binary_mask, (512, 512), cv2.INTER_NEAREST)
    if len(binary_mask.shape) == 2:
        binary_mask = binary_mask[..., np.newaxis]
    return binary_mask


def create_mask(args):
    prediction, annotation = args
    bbox_prediction, mask_prediction = prediction

    samples = []
    metrics = []
    for cls, (masks, bboxes) in enumerate(zip(mask_prediction, bbox_prediction)):
        if masks:
            prediction_mask = decode_and_resize(
                masks=masks,
                x_min=annotation.get('x_min', None),
                x_max=annotation.get('x_max', None),
                y_min=annotation.get('y_min', None),
                y_max=annotation.get('y_max', None),
                original_height=annotation.get('original_height', None),
                original_width=annotation.get('original_width', None)
            )
            if not check_overlaps(prediction_mask):
                prediction_mask = hard_overlaps_suppression(prediction_mask.astype(bool), bboxes[..., -1])

            if annotation['ann']['masks'] is not None:
                indices = np.where(annotation['ann']['labels'] - 1 == cls)[0]
                if len(indices):
                    true_mask = decode_and_resize([annotation['ann']['masks'][i] for i in indices])
                    true_labeled_mask = create_labeled_mask(true_mask)
                    prediction_labeled_mask = create_labeled_mask(prediction_mask)
                    metrics.append(calc_score_per_class(true_labeled_mask, prediction_labeled_mask))
                else:
                    metrics.append(0)

            for mask_id in range(prediction_mask.shape[-1]):
                rle = kaggle_rle_encode(prediction_mask[..., mask_id])
                samples.append(
                    {
                        'ImageId': annotation['filename'],
                        'EncodedPixels': ' '.join(map(str, rle)),
                        'ClassId': str(cls)
                    }
                )
        elif annotation['ann']['masks'] is not None and np.any(annotation['ann']['labels'] - 1 == cls):
            metrics.append(0)

    if not len(samples):
        samples.append({'ImageId': annotation['filename'], 'EncodedPixels': '', 'ClassId': '23'})
    if not len(metrics):
        metrics.append(1)
    return {'samples': samples, 'metric': np.mean(metrics)}


def main():
    args = parse_args()
    predictions = mmcv.load(args.predictions)
    annotation = mmcv.load(args.annotation)
    print(f'predictions: {args.predictions}')
    print(f'output: {args.output}')

    with Pool(args.n_jobs) as p:
        results = list(
            tqdm(iterable=p.imap_unordered(create_mask, zip(predictions, annotation)), total=len(predictions))
        )
    samples = sum([x['samples'] for x in results], [])
    metrics = [x['metric'] for x in results]

    submission = pd.DataFrame(samples)
    if args.add_metric:
        submission['mAP'] = sum([[x['metric']] * len(x['samples']) for x in results], [])

    submission.to_csv(args.output, index=False)
    print(f'Mask mAP {np.mean(metrics)}')


if __name__ == '__main__':
    main()
