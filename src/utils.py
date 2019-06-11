import numpy as np
from pycocotools import mask as mutils

from src.rle import kaggle2coco


def group2mmdetection(group: dict) -> dict:
    image_id, group = group
    assert group['Width'].max() == group['Width'].min()
    assert group['Height'].max() == group['Height'].min()
    height, width = group['Height'].max(), group['Width'].max()
    rles = group['EncodedPixels'].apply(lambda x: kaggle2coco(list(map(int, x.split())), height, width)).tolist()
    rles = mutils.frPyObjects(rles, height, width)
    masks = mutils.decode(rles)
    bboxes = mutils.toBbox(mutils.encode(np.asfortranarray(masks.astype(np.uint8))))
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return {
        'filename': image_id,
        'width': width,
        'height': height,
        'ann':
            {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'original_labels': group['ClassId'].values,
                'labels': group['ClassId'].apply(lambda x: x.split('_')[0]).values.astype(np.int) + 1,
                'masks': rles
            }
    }


def create_labeled_mask(mask):
    return (np.arange(1, mask.shape[-1] + 1)[None, None, :] * mask).sum(-1)


def check_overlaps(mask):
    overlap_mask = mask.sum(axis=-1)
    return np.array_equal(overlap_mask, overlap_mask.astype(bool))


def hard_overlaps_suppression(binary_mask, scores):
    not_overlap_mask = []
    for i in np.argsort(scores)[::-1]:
        current_mask = binary_mask[..., i].copy()
        for mask in not_overlap_mask:
            current_mask = np.bitwise_and(current_mask, np.invert(mask))
        not_overlap_mask.append(current_mask)
    return np.stack(not_overlap_mask, -1)