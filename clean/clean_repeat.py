import pickle
import sys

import numpy as np
from sklearn import preprocessing

threshold = 0.99


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
    FEAT_MATRIX, IMAGE_NAME_LIST = process_info(feat_info)

    #
    # for image_name in IMAGE_NAME_LIST:
    #     print(image_name)

    ID_NUM_DICT = {
        '1055': 610, '0767': 1018,
        '0968': 118, '0760': 122, '1374': 731, '0174': 306,
        '0383': 774, '1161': 113, '1477': 864, '1273': 223,
        '1350': 291, '0651': 188, '0514': 122, '0112': 150
    }

    # count and check
    count_dict = {}
    for item in IMAGE_NAME_LIST:
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

    # get image list
    id_image_list_dict = {}
    for idx, image_name in enumerate(IMAGE_NAME_LIST):
        # 0057_c1_928644343.png
        pid = image_name.split('_')[0]
        if pid not in ID_NUM_DICT.keys():
            continue
        else:
            if pid not in id_image_list_dict.keys():
                id_image_list_dict[pid] = [image_name]
            else:
                id_image_list_dict[pid].append(image_name)

    # for pid in ID_NUM_DICT.keys():
    #     print(pid)
    # for pid, id_image_list in id_image_list_dict.items():
    #     print(pid, id_image_list)

    # going through all the pids
    clean_id_set = set()
    dirty_id_set = set()
    for pid, id_image_list in id_image_list_dict.items():

        pid_all_feats = []

        #
        # all_feats
        for id_image in id_image_list:
            # id_image 0057_c1_928644343.png
            pid_all_feats.append(FEAT_MATRIX[IMAGE_NAME_LIST.index(id_image)])

        for id_image in id_image_list:
            # id_image 0057_c1_928644343.png
            curr_feat = FEAT_MATRIX[IMAGE_NAME_LIST.index(id_image)]
            curr_feat = curr_feat.reshape(1, -1)
            #
            curr_distance_matrix = np.dot(curr_feat, np.array(pid_all_feats).T)

            repeat_num = np.sum(curr_distance_matrix > threshold)

            if repeat_num > 3 and id_image not in dirty_id_set:
                #
                clean_id_set.add(id_image)

                #
                id_image_array = np.array(id_image_list)
                dirty_image_array = id_image_array[np.reshape(curr_distance_matrix > threshold, -1)]

                res = []
                for dirty_image in dirty_image_array:
                    if dirty_image not in clean_id_set:
                        dirty_id_set.add(dirty_image)
                        res.append(dirty_image)

                with open('/home/xiangan/dgreid/clean/dirty_0.99/%s' % id_image, 'w') as f:
                    f.write("\n".join(res))

            #
    for i in dirty_id_set:
        print(i)


if __name__ == '__main__':
    main()
