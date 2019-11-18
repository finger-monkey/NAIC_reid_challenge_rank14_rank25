import pickle
import argparse
import numpy as np
from sklearn import preprocessing
import json


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
    #
    QUERY_FEATURE_PATH = '/home/xiangan/dgreid/features/024/query_a_feature.feat'
    #
    THRESHOLD = 0.9

    query_info = pickle.load(open(QUERY_FEATURE_PATH, 'rb'))
    query_feats, query_imgnames = process_info(query_info)
    query_sim = np.dot(query_feats, query_feats.T)

    count = 0
    for idx, distance_arr in enumerate(query_sim):
        if sum(distance_arr > THRESHOLD):
            count += 1
        # print(sum(distance_arr > THRESHOLD))

    print(count)
    # print(query_imgnames)
    print(len(query_imgnames))


if __name__ == '__main__':
    main()