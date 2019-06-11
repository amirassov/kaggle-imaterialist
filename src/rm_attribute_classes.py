import argparse
import pandas as pd

ATTRIBUTE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    submission = pd.read_csv(args.submission)
    submission_without_attributes = submission[~submission['ClassId'].isin(ATTRIBUTE_CLASSES)].copy()
    empty_ids = list(set(submission['ImageId']) - set(submission_without_attributes['ImageId']))
    submission_empty = pd.DataFrame([empty_ids, [''] * len(empty_ids), ['23'] * len(empty_ids)]
                                   ).T.rename(columns={
                                       0: 'ImageId',
                                       1: 'EncodedPixels',
                                       2: 'ClassId'
                                   })
    pd.concat([submission_without_attributes, submission_empty], sort=True).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
