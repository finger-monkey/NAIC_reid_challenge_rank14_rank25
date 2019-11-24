import os
import pickle

import numpy as np
from sklearn import preprocessing
from tqdm import tqdm


def process_info(info):
    feats = []
    imgnames = []
    for i in range(len(info)):
        feats.append(info[i][0].flatten())
        imgnames.append(info[i][1])
    feats = np.array(feats)
    feats = preprocessing.normalize(feats)
    return feats, imgnames


def main():
    # path
    feature_path = "/home/xiangan/dgreid/features/trainset_features_056/trainset.feat"
    # info
    feat_info = pickle.load(open(feature_path, 'rb'))
    # feats and names
    feats, image_name_list = process_info(feat_info)

    for image_name in image_name_list:
        print(image_name)


if __name__ == '__main__':
    main()