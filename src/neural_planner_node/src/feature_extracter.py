#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2017.1.22
"""This extractor is working for build sift descriptor( not keypoint)."""

import math
import operator
import numpy
import cv2
import pdb

SAMPLE_NUM = 100   # extract 100 sample
SAMPLE_SIZE = 128  # sift descriptor has 128 byte
SAMPLE_DEPTH = 1   # default parameter, just 1
SIFT_DETECTOR = cv2.xfeatures2d.SIFT_create()


def _grid_feature(kps, des, img_size, grid_shape, sample_num):
    """Build grid feature from sift descriptor feature.
    Args:
        kp: sift keypoints.
        des: sift descriptor.
        img_size: image size, a 1D list, [heigh,width]/[rows,cols].
        grid_shape: [heigh,width]/[rows,cols] for grid feature.
    return:
        grid feature rect with shape [sample_num,SAMPLE_SIZE]
    """
    global SAMPLE_SIZE

    # Create bucket for every grid
    grid_buckets = [[{} for i in range(grid_shape[1])] for i in range(grid_shape[0])]

    cell_h = img_size[0]/grid_shape[0] # heigh for every cell
    cell_w = img_size[1]/grid_shape[1]
    for idx, kp in enumerate(kps):
        pos_row = int(kp.pt[1]/cell_h) # kp.pt[0] is x-coordinate, it's column/width.
        pos_col = int(kp.pt[0]/cell_w)
        grid_buckets[pos_row][pos_col][idx] = kp.response

    # Sort every bucket
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            bucket = grid_buckets[i][j]
            grid_buckets[i][j] = sorted(bucket.items(), key=operator.itemgetter(1), reverse=True)

    # Sample according to ranking
    max_sample_num = sample_num + grid_shape[0]*grid_shape[1]
    g_feature = numpy.zeros([max_sample_num, SAMPLE_SIZE])
    cur_num = 0 # current number
    # 最大可能采样到100层，大部分情况会提前完成采样，少部分情况会样本不足。样本不足情况下，则正常退出即可
    for rank_num in range(sample_num):
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                bucket = grid_buckets[i][j]
                if len(bucket) > rank_num: # 判断桶里面是否有当前排名的数据
                    idx = bucket[rank_num][0]
                    g_feature[cur_num, :] = des[idx, :]
                    cur_num = cur_num + 1
                    # Copy des[idx,:] to new array
        if  cur_num >= sample_num: # 提前完成采样
            # Finish sample
            break

    # truncation array to sample_num
    g_feature = g_feature[0:sample_num, :]
    return grid_buckets, g_feature


def _sift(gray_image):
    """Extract sift descriptor for a gray image.
    Args:
      image: this is a 2d gray image.
    Return:
        sift descriptor.
    """
    global SIFT_DETECTOR
    # image must be a 2D numpy array.
    kps = SIFT_DETECTOR.detect(gray_image, None)
    kps, des = SIFT_DETECTOR.compute(gray_image, kps)

    return kps, des


def grid_feature(image, grid_shape):
    """Build grid feature for a image.
    Args:
        image: a 2-D array.
        grid_shape: this is a 1D list, [heigh,width]/[rows,cols].
    return:
        grid feature with shape [SAMPLE_NUM,SAMPLE_SIZE].
    """
    global SAMPLE_NUM

    assert len(image.shape) == 2, (\
'Error: input image must be a 2D gray image.')
    assert len(grid_shape) == 2, (\
'Error: output dims must be [heigh,width].')
    kps, des = _sift(image)
    grid_buckets, g_feature = _grid_feature(kps, des, image.shape, grid_shape, SAMPLE_NUM)

    return grid_buckets, g_feature


def feature_extracter(image, grid_shape):
    """Build grid feature for a image.
    Args:
        images: 2-D array, [heigh, width].
        dims: [heigh, width] for grid feature.
    Return:
        a grid buckets which is used to region-ranking.
        a grid feature with shape [SAMPLE_NUM,SAMPLE_SIZE].
    """
    global SAMPLE_NUM
    global SAMPLE_SIZE
    global SAMPLE_DEPTH

    assert len(image.shape) == 2, "Input image must be 2D"
    grid_buckets, g_feature = grid_feature(image, grid_shape)
    return grid_buckets, g_feature


def features_extracter(images, grid_shape):
    """Build grid feature for a series of gray images.
    Args:
        images: 4-D array, [batch,heigh,width,channels=1].
        grid_shape: [num_in_rows,num_in_cols]/[heigh,width] for grid feature.
    Return:
        a array of grid bucket, type is list, shape is [batch,grid_bucket]
        a array of grid feature, type is numpy.array, [batch,features,descriptors=128,channels=1].
    """
    global SAMPLE_NUM
    global SAMPLE_SIZE
    global SAMPLE_DEPTH

    assert len(images.shape) == 4, "Input images must be 4D."
    assert images.shape[3] == 1, "Input images must be gray image."
    assert len(grid_shape) == 2, "grid_shape must be 2D."

    image_num = images.shape[0]
    gf_dims = [image_num, SAMPLE_NUM, SAMPLE_SIZE, SAMPLE_DEPTH]
    features = numpy.zeros(gf_dims)
    buckets = {}
    for i in range(image_num):
        image = images[i, :, :, 0]
        grid_bucket, g_feature = grid_feature(image, grid_shape)
        features[i, :, :, 0] = g_feature
        buckets[i] = grid_bucket
    return buckets, features


"""The following part is used to test this model."""
def show_sift(canvas):
    """
    """
    kps = []
    kp = cv2.KeyPoint(30, 30, 100, 10)
    kps.append(kp)
    cv2.drawKeypoints(canvas, kps, canvas)
    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)


def show_grid_bucket(grid_buckets, img, canvas=None):
    """Show a grid feature.
    Args:
        grid_buckets: 2D array. [grid_row,grid_col]. Every obj is a list, too.
        img: origial image which bucket from.
        canvas: 3D array, [heigh,width,depth]. color image(werid?).
    """
    global SAMPLE_NUM
    global SAMPLE_SIZE
    # Transform grid_bucket to keypoints
    kps = SIFT_DETECTOR.detect(img, None)
    grid_kps = []
    grid_row = len(grid_buckets)
    grid_col = len(grid_buckets[0])
    # Sample according to ranking
    sample_num = SAMPLE_NUM
    cur_num = 0 # current number
    # 注意，下面这个采集方法应该与_grid_feature中保持一致，否则显示到的数据可能不是真实数据
    # 最大可能采样到100层，大部分情况会提前完成采样，少部分情况会样本不足。样本不足情况下，则正常退出即可
    for rank_num in range(sample_num):
        for i in range(grid_row):
            for j in range(grid_col):
                bucket = grid_buckets[i][j]
                if len(bucket) > rank_num: # 判断桶里面是否有当前排名的数据
                    idx = bucket[rank_num][0]
                    grid_kps.append(kps[idx])
                    cur_num = cur_num + 1
                    # Copy des[idx,:] to new array
        if  cur_num >= sample_num: # 提前完成采样
            # truncation array to sample_num
            grid_kps = grid_kps[0:sample_num]
            # Finish sample
            break
    # Draw grid keypointt
    cv2.drawKeypoints(canvas, grid_kps, canvas)
    cv2.namedWindow('canvas')
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)


def main():
    """This part is just a test program for this library.
    """
    # Load image
    img = cv2.imread('/home/blue/lab/robot/omnidirectional_vehicle/neural_planner_V2.0/src/neural_planner_node/src/test2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs = numpy.zeros([1, gray.shape[0], gray.shape[1], 1], dtype='uint8')
    imgs[0, :, :, 0] = gray
    #cv2.namedWindow('original')
    #cv2.imshow('original', imgs[0, :, :, 0])

    buckets, features = features_extracter(imgs, [3, 3])
    show_grid_bucket(buckets[0], imgs[0, :, :, 0], canvas=img)


if __name__ == '__main__':
    main()
