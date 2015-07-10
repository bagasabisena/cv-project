from __future__ import division

import cv2
from PIL import Image
import numpy as np
import math
import scipy.stats
import os
import re
import utils
import lbp
import normalization
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_lfw_people

# extract non uniform LBP feature for YALE dataset
def lbp_feature():

    base_path = '../data/YALE/centered/'
    images = os.listdir(base_path)
    labels = np.empty(len(images))
    data = np.empty([len(images), 5120])

    # parameter for LBP
    face_area = (15, 215, 15, 175)
    block_size = (40, 40)
    point = 8
    radius = 2

    for i, im_name in enumerate(images):
        print 'extracting image %d out of %d' % (i+1, len(images),)
        label = int(re.search('\d+', im_name).group())
        im_path = base_path + im_name
        im = utils.to_numpy(Image.open(im_path))
        lbp_hist = lbp.grid_lbp(im, point, radius, face_area, block_size)
        feat = lbp_hist.reshape(np.size(lbp_hist))
        data[i, :] = feat
        labels[i] = label

    print 'saving data'
    np.save('../data/dataset/lbp_data.npy', data)
    np.save('../data/dataset/label.npy', labels)


# extract uniform pattern LBP feature for YALE dataset
def lbp_uni_feature(norm=True, filename='dataset.npy'):
    base_path = '../data/YALE/centered/'
    images = os.listdir(base_path)
    labels = np.empty(len(images))

    # parameter for uniform LBP
    face_area = (15, 215, 15, 175)
    block_size = (10, 10)
    radius = 2
    min_height = face_area[0]
    max_height = face_area[1]
    min_width = face_area[2]
    max_width = face_area[3]
    block_height = block_size[0]
    block_width = block_size[1]
    block_total = ((max_height - min_height) / block_height) * ((max_width - min_width) / block_width)
    data = np.empty([len(images), 59*block_total])

    for i, im_name in enumerate(images):
        print 'extracting image %d out of %d' % (i+1, len(images),)
        label = int(re.search('\d+', im_name).group())
        im_path = base_path + im_name
        im = utils.to_numpy(Image.open(im_path))
        if norm:
            im = normalization.tan_triggs_norm(im, 0.2, 1, 4, 0.1, 10)

        lbp_hist = lbp.uniform_lbp2(im, radius, face_area, block_size)
        feat = lbp_hist.reshape(np.size(lbp_hist))
        data[i, :] = feat
        labels[i] = label

    print 'saving data'
    np.save('../data/dataset/%s' % filename, data)


# extract eigenfaces for YALE dataset
def eigenface_feature(n_components = 10, filename='lbp_pca.npy'):
    base_path = '../data/YALE/centered/'
    images = os.listdir(base_path)
    labels = np.empty(len(images))
    h = 200
    w = 160
    data = np.empty([len(images), h*w]) # size of cropped image

    # cropped the face area
    face_area = (15, 215, 15, 175)

    for i, im_name in enumerate(images):
        print 'extracting image %d out of %d' % (i+1, len(images),)
        label = int(re.search('\d+', im_name).group())
        im_path = base_path + im_name
        im = utils.to_numpy(Image.open(im_path))
        im = im[15:215, 15:175]
        feat = im.reshape(np.size(im))
        data[i, :] = feat
        labels[i] = label

    # compute the pca
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    data_pca = pca.transform(data)

    print 'saving data'
    np.save('../data/dataset/%s' % filename, data_pca)
    np.save('../data/dataset/eigenfaces.npy', eigenfaces)


# get label for YALE dataset
def save_label():
    base_path = '../data/YALE/centered/'
    images = os.listdir(base_path)
    labels = np.empty(len(images))

    for i, im_name in enumerate(images):
        label = int(re.search('\d+', im_name).group())
        labels[i] = label

    print 'saving data'
    np.save('../data/dataset/label.npy', labels)


# extract uniform pattern LBP feature for LFW dataset
def lfw_lbp(norm=True, filename='dataset.npy', min_people=20):

    # use helper from scikit-learn to prepare the LFW dataset
    print "prepare LFW image"
    print "requires download ~200MB for the first time"
    print "Next use will not download again"
    lfw_people = fetch_lfw_people(data_home='../data/', min_faces_per_person=20)
    images = lfw_people.images
    label = lfw_people.target

    # parameter for uniform LBP
    face_area = (1, 61, 1, 46)
    block_size = (5, 5)
    radius = 2
    min_height = face_area[0]
    max_height = face_area[1]
    min_width = face_area[2]
    max_width = face_area[3]
    block_height = block_size[0]
    block_width = block_size[1]
    block_total = ((max_height - min_height) / block_height) * ((max_width - min_width) / block_width)
    data = np.empty([len(images), 59*block_total])

    for i, im in enumerate(images):
        print 'extracting image %d out of %d' % (i+1, len(images),)
        if norm:
            im = normalization.tan_triggs_norm(im, 0.2, 1, 4, 0.1, 10)

        lbp_hist = lbp.uniform_lbp2(im, radius, face_area, block_size)
        feat = lbp_hist.reshape(np.size(lbp_hist))
        data[i, :] = feat

    print 'saving data'
    np.save('../data/dataset/%s' % filename, data)
    np.save('../data/dataset/%s_label' % filename, label)


# extract label for LFW
def lfw_label(filename='lfw_label', min_people=20):
    print "prepare LFW label"
    lfw_people = fetch_lfw_people(min_faces_per_person=min_people)
    label = lfw_people.target
    np.save('../data/dataset/%s' % filename, label)


# extract eigenfaces for LFW dataset
def lfw_eigenface(filename='lfw_pca.npy', n_components=10, min_people=20):
    # use helper from scikit-learn to prepare the LFW dataset
    print "prepare LFW image"
    lfw_people = fetch_lfw_people(data_home='../data/', min_faces_per_person=20)
    images = lfw_people.images
    h = 60
    w = 45
    data = np.empty([len(images), h*w]) # size of cropped image

    # cropped the face area
    face_area = (1, 61, 1, 46)

    for i, im in enumerate(images):
        print 'extracting image %d out of %d' % (i+1, len(images),)
        im = im[face_area[0]:face_area[1], face_area[2]:face_area[3]]
        feat = im.reshape(np.size(im))
        data[i, :] = feat

    # compute the pca
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    data_pca = pca.transform(data)

    print 'saving data'
    np.save('../data/dataset/%s' % filename, data_pca)
    np.save('../data/dataset/%s_eigen' % filename, eigenfaces)


if __name__ == '__main__':
    #lfw_lbp(False, 'lfw_lbp_nonorm.npy', 20)
    #lfw_lbp(True, 'lfw_lbp.npy', 20)
    lfw_eigenface(n_components=100, min_people=20)
