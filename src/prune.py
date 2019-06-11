import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = torch.load(args.weights)
    weights['state_dict'] = {
        k: v
        for k, v in weights['state_dict'].items()
        if not k.startswith('bbox_head') and not k.startswith('mask_head')
    }
    torch.save(weights, args.output)


if __name__ == '__main__':
    main()
