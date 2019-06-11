import argparse
import os
import pickle
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from src.utils import group2mmdetection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_jobs', type=int, default=80)
    parser.add_argument('--n_samples', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    annotation = pd.read_csv(args.annotation)
    files = sorted(os.listdir(args.root))
    if args.n_samples != -1:
        files = files[:args.n_samples]
    annotation = annotation.loc[annotation['ImageId'].isin(set(files))]
    print(len(annotation), len(set(annotation['ImageId'])))

    groups = list(annotation.groupby('ImageId'))

    with Pool(args.n_jobs) as p:
        samples = list(tqdm(iterable=p.imap_unordered(group2mmdetection, groups), total=len(groups)))

    with open(args.output, 'wb') as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    main()
