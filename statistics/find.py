import pickle

import numpy as np
import os
from sklearn import preprocessing
from tqdm import tqdm
import json


def process_info(info):
    """
    get features and normalizing features.
    Args:
        info:

    Returns:

    """
    feats = []
    imgnames = []
    for i in range(len(info)):
        feats.append(info[i][0].flatten())
        imgnames.append(info[i][1])
    feats = np.array(feats)
    feats = preprocessing.normalize(feats)
    return feats, imgnames


def main():
    query = "395592518.png"

    gallery_info = pickle.load(open('/home/xiangan/reid_features/gallery.feat', 'rb'))
    query_info = pickle.load(open('/home/xiangan/reid_features/query.feat', 'rb'))
    train_info = pickle.load(open('/home/xiangan/reid_features/train.feat', 'rb'))

    gallery_feats, gallery_imgnames = process_info(gallery_info)
    query_feats, query_imgnames = process_info(query_info)
    train_feats, train_imgnames = process_info(train_info)

    all_feat = np.concatenate((query_feats, train_feats, gallery_feats))
    all_name = query_imgnames + train_imgnames + gallery_imgnames

    for i in all_name:
        if query in i:
            print(i)

    feat = all_feat[all_name.index(query)]

    sim = np.dot(feat, all_feat.T)
    indices = np.argsort(-sim, axis=1)

    order = indices[0][:50]
    query_gallery = []
    for gallery_index in order:
        query_gallery.append(all_name[gallery_index])
    process_info(query_gallery)


if __name__ == '__main__':
    main()
