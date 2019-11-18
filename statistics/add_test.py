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
    #
    QUERY_FEATURE_PATH = '/home/xiangan/dgreid/features/024/query_a_feature.feat'
    GALLERY_FEATURE_PATH = '/home/xiangan/dgreid/features/024/gallery_a_feature.feat'
    OUTPUT_PATH = '/home/xiangan/code_and_data/train_split/test_extra'
    TEST_QUERY_PATH = '/home/xiangan/data_reid/testA/query_a'
    #
    THRESHOLD = 0.6

    query_info = pickle.load(open(QUERY_FEATURE_PATH, 'rb'))
    query_feats, query_imgnames = process_info(query_info)
    # gallery_info = pickle.load(open(GALLERY_FEATURE_PATH, 'rb'))
    # gallery_feats, gallery_imgnames = process_info(gallery_info)

    query_sim = np.dot(query_feats, query_feats.T)
    #
    # sim = np.dot(query_feats, gallery_feats.T)
    # num_q, num_g = sim.shape
    # indices = np.argsort(-sim, axis=1)

    count = 0
    # 0.9 0
    # 0.7 17
    # 0.6 160
    # 0.5 789
    dirty_id_set = set()
    clean_id_set = set()
    for idx, distance_arr in enumerate(query_sim):
        if sum(distance_arr > THRESHOLD) > 1:
            count += 1
            dirty_id_set.add(query_imgnames[idx])
        else:
            clean_id_set.add(query_imgnames[idx])

    # print(count)
    # print(query_imgnames)
    # print(len(query_imgnames))
    print(len(dirty_id_set))
    print(len(clean_id_set))
    count = 4768
    for clean_name in tqdm(clean_id_set):
        input_path = os.path.join(TEST_QUERY_PATH, clean_name)
        output_name = os.path.join(OUTPUT_PATH, "%d_c1_%s" % (count, clean_name))
        open(output_name, 'wb').write(open(input_path, 'rb').read())
        count += 1


if __name__ == '__main__':
    main()
