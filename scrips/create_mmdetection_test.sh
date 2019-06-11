#!/usr/bin/env bash

PYTHONPATH=/kaggle-imaterialist python /kaggle-imaterialist/src/create_mmdetection_test.py \
    --annotation=/data/sample_submission.csv \
    --root=/data/test \
    --output=/data/test_mmdetection.pkl
