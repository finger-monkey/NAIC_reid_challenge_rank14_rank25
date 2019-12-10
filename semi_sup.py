import pickle

import numpy as np
import os
from sklearn import preprocessing
from tqdm import tqdm
import json
import copy


def process_info(info):
    feats, imgnames = info
    feats = preprocessing.normalize(feats)
    return feats, imgnames


def get_clean_query(query_feats, query_imgnames, threshold):
    # score matrix
    query_sim = np.dot(query_feats, query_feats.T)
    #
    count = 0
    # 0.9 0
    # 0.7 17
    # 0.6 160
    # 0.5 789
    dirty_id_set = set()
    clean_id_set = set()
    for idx, distance_arr in enumerate(query_sim):
        if sum(distance_arr > threshold) > 1:
            count += 1
            dirty_id_set.add(query_imgnames[idx])
        else:
            clean_id_set.add(query_imgnames[idx])

    print('dirty id num is:  ', len(dirty_id_set))
    print('clean id num is:  ', len(clean_id_set))
    return clean_id_set


def main():
    # threshold
    rank_dirty_threshold = 0.7
    query_dirty_threshold = 0.6
    #
    # rank_list
    testA_ranklist = '/home/xiangan/dgreid/rerank_all001.json'
    # testA
    testA_gallery_info = pickle.load(open('/home/xiangan/dgreid/features/all001/gallery_feature.feat', 'rb'))
    testA_query_info = pickle.load(open('/home/xiangan/dgreid/features/all001/query_feature.feat', 'rb'))

    testA_gallery_feats, testA_gallery_imgnames = process_info(testA_gallery_info)
    testA_query_feats, testA_query_imgnames = process_info(testA_query_info)

    # assert testA_gallery_feats.shape[0] == 5366
    # assert testA_query_feats.shape[0] == 1348

    QUERY_FEAT = testA_query_feats
    IMAGE_NAME = copy.deepcopy(testA_query_imgnames)

    clean_query_id_set = get_clean_query(QUERY_FEAT, IMAGE_NAME, query_dirty_threshold)

    f = open(testA_ranklist, encoding='utf-8')
    content = f.read()
    dic = json.loads(content)
    cleaned_rank_dict_testA = {}
    for query_name in list(dic.keys()):
        # print(query_name)
        if query_name in clean_query_id_set:
            origin_ranklist = dic[query_name][:10]
            query_cur_feat = testA_query_feats[testA_query_imgnames.index(query_name)]

            cleaned_ranklist = []
            cleaned_count = 0
            for gallery_name in origin_ranklist:
                rank_cur_feat = testA_gallery_feats[testA_gallery_imgnames.index(gallery_name)]
                score = np.dot(query_cur_feat, rank_cur_feat)
                if score > rank_dirty_threshold:
                    cleaned_ranklist.append(gallery_name)
                    cleaned_count += 1
                else:
                    break
            cleaned_rank_dict_testA[query_name] = cleaned_ranklist
            # print(cleaned_count)

    count = 9968
    for query, clean_list in tqdm(cleaned_rank_dict_testA.items()):
        input_path = os.path.join("/data/xiangan/reid_final/test/query_a", query)
        output_name = os.path.join("/data/xiangan/reid_final/extra_1",
                                   "%d_c1_%s" % (count, query))
        open(output_name, 'wb').write(open(input_path, 'rb').read())
        for clean_name in clean_list:
            input_path = os.path.join("/data/xiangan/reid_final/test/gallery_a", clean_name)
            output_name = os.path.join("/data/xiangan/reid_final/extra_1",
                                       "%d_c1_%s" % (count, clean_name))
            open(output_name, 'wb').write(open(input_path, 'rb').read())

        count += 1


if __name__ == '__main__':
    main()
