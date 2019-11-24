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

    #
    # for image_name in image_name_list:
    #     print(image_name)

    ID_NUM_DICT = {
        '1055': 610, '0767': 1018,
        '0968': 118, '0760': 122, '1374': 731, '0174': 306,
        '0383': 774, '1161': 113, '1477': 864, '1273': 223,
        '1350': 291, '0651': 188, '0514': 122, '0112': 150
    }

    count_dict = {}
    for item in image_name_list:
        # 0057_c1_928644343.png
        pid = item.split('_')[0]

        # count
        if pid not in count_dict:
            count_dict[pid] = 1
        else:
            count_dict[pid] += 1

    # check
    for pid, count in ID_NUM_DICT.items():
        assert count_dict[pid] == count

    for pid in ID_NUM_DICT.keys():
        print(pid)


if __name__ == '__main__':
    main()
