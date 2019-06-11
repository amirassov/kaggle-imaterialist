import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool

import jpeg4py as jpeg
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_jobs', type=int, default=40)
    return parser.parse_args()


def convert(group: dict, root) -> dict:
    image_id, group = group
    image = jpeg.JPEG(os.path.join(root, image_id)).decode()
    height, width = image.shape[:2]
    return {
        'filename': image_id,
        'width': width,
        'height': height,
        'ann': {
            'bboxes': None,
            'labels': None,
            'masks': None
        }
    }


def main():
    args = parse_args()
    annotation = pd.read_csv(args.annotation)
    print(len(annotation), len(set(annotation['ImageId'])))

    groups = list(annotation.groupby('ImageId'))

    with Pool(args.n_jobs) as p:
        samples = list(tqdm(iterable=p.imap_unordered(partial(convert, root=args.root), groups), total=len(groups)))

    with open(args.output, 'wb') as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    main()
