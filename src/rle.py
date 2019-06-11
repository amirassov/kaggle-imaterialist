from itertools import groupby
from pycocotools import mask as mutils
import numpy as np
from tqdm import tqdm


def kaggle_rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 1
    rle[1::2] -= rle[::2]
    return rle.tolist()


def kaggle_rle_decode(rle, h, w):
    starts, lengths = map(np.asarray, (rle[::2], rle[1::2]))
    starts -= 1
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((w, h)).T


def coco_rle_encode(mask):
    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def coco_rle_decode(rle, h, w):
    return mutils.decode(mutils.frPyObjects(rle, h, w))


def kaggle2coco(kaggle_rle, h, w):
    if not len(kaggle_rle):
        return {'counts': [h * w], 'size': [h, w]}
    roll2 = np.roll(kaggle_rle, 2)
    roll2[:2] = 1

    roll1 = np.roll(kaggle_rle, 1)
    roll1[:1] = 0

    if h * w != kaggle_rle[-1] + kaggle_rle[-2] - 1:
        shift = 1
        end_value = h * w - kaggle_rle[-1] - kaggle_rle[-2] + 1
    else:
        shift = 0
        end_value = 0
    coco_rle = np.full(len(kaggle_rle) + shift, end_value)
    coco_rle[:len(coco_rle) - shift] = kaggle_rle.copy()
    coco_rle[:len(coco_rle) - shift:2] = (kaggle_rle - roll1 - roll2)[::2].copy()
    return {'counts': coco_rle.tolist(), 'size': [h, w]}


def main():
    for _ in tqdm(range(100)):
        h = np.random.randint(1, 1000)
        w = np.random.randint(1, 1000)
        mask = np.random.randint(0, 2, h * w).reshape(h, w)

        kaggle_rle = kaggle_rle_encode(mask)
        coco_rle = coco_rle_encode(mask)
        assert coco_rle == kaggle2coco(kaggle_rle, h, w)
        assert np.all(mask == kaggle_rle_decode(kaggle_rle, h, w))
        assert np.all(mask == coco_rle_decode(coco_rle, h, w))


if __name__ == '__main__':
    main()
