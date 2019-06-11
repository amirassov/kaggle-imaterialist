import argparse
import mmcv
import numpy as np
import pickle

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str)
    parser.add_argument('--train_output', type=str)
    parser.add_argument('--val_output', type=str)
    parser.add_argument('--n_splits', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=40)
    parser.add_argument('--n_samples', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    annotation = mmcv.load(args.annotation)
    all_labels = [x['ann']['labels'] for x in annotation]
    y = np.zeros((len(all_labels), max([max(x) for x in all_labels])))
    for labels in all_labels:
        y[:, labels - 1] = 1

    mskf = MultilabelStratifiedKFold(n_splits=args.n_splits, random_state=777)

    for train_index, val_index in mskf.split(y, y):
        train_annotation = [x for i, x in enumerate(annotation) if i in train_index]
        val_annotation = [x for i, x in enumerate(annotation) if i in val_index]
        with open(args.train_output, 'wb') as f:
            pickle.dump(train_annotation, f)
        with open(args.val_output, 'wb') as f:
            pickle.dump(val_annotation, f)
        print(f'train size: {len(train_annotation)}, val size: {len(val_annotation)}')
        break


if __name__ == '__main__':
    main()
