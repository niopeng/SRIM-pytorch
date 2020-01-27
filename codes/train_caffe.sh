#!/bin/bash
python train_caffe_two.py -opt options/train/train_caffe_monarch_x4.json
python train_caffe_three.py -opt options/train/train_caffe_monarch_x8.json
