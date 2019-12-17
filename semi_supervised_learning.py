import pickle

import numpy as np
import os
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import json
import copy


def process_info(info):
    """"""
    feats, img_names = info
    feats = preprocessing.normalize(feats)
    return feats, img_names


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


def visualization_a(rank_dict, test_root, output_root):
    query_root = os.path.join(test_root, 'query_a')
    gallery_root = os.path.join(test_root, 'gallery_a')
    for query in rank_dict.keys():
        query_folder = os.path.join(output_root, "%d_%s" % (len(rank_dict[query]), query))
        os.makedirs(query_folder)
        open(os.path.join(query_folder, query), 'wb').write(open(os.path.join(query_root, query), 'rb').read())
        for ranid, neighbor in enumerate(rank_dict[query]):
            target_path = os.path.join(query_folder, "%d_%s" % (ranid + 1, neighbor))
            source_path = os.path.join(gallery_root, neighbor)
            open(target_path, 'wb').write(open(source_path, 'rb').read())


def main():
    # threshold
    rank_dirty_threshold = 0.7
    query_dirty_threshold = 0.6
    #
    # testA rank_list
    testA_ranklist = 'results/rerank_503_testA.json'
    # testB rank_list
    testB_ranklist = 'results/rerank_503_testB.json'
    # testA
    testA_gallery_info = pickle.load(open(
        '/home/xiangan/dgreid/features/jiankang_train_001/gallery_feature.feat', 'rb')
    )
    testA_query_info = pickle.load(open(
        '/home/xiangan/dgreid/features/jiankang_train_001/query_feature.feat', 'rb')
    )

    testA_gallery_feats, testA_gallery_img_names = process_info(testA_gallery_info)
    testA_query_feats, testA_query_img_names = process_info(testA_query_info)

    cls = AgglomerativeClustering(
        n_clusters=None,
        linkage='average',
        affinity="cosine",
        distance_threshold=0.8
    )

    cls.fit(testA_query_feats)
    print(len(set(cls.labels_)))

    # # testB
    # testB_gallery_info = pickle.load(open('/home/xiangan/dgreid/features_testB/503/gallery_b_feature.feat', 'rb'))
    # testB_query_info = pickle.load(open('/home/xiangan/dgreid/features_testB/503/query_b_feature.feat', 'rb'))
    #
    # testB_gallery_feats, testB_gallery_imgnames = process_info(testB_gallery_info)
    # testB_query_feats, testB_query_imgnames = process_info(testB_query_info)
    #
    # assert testA_gallery_feats.shape[0] == 5366
    # assert testA_query_feats.shape[0] == 1348
    # assert testB_gallery_feats.shape[0] == 13099
    # assert testB_query_feats.shape[0] == 3147
    #
    # QUERY_FEAT = np.concatenate((testA_query_feats, testB_query_feats), axis=0)
    # IMAGE_NAME = copy.deepcopy(testA_query_imgnames)
    # IMAGE_NAME.extend(testB_query_imgnames)
    # clean_query_id_set = get_clean_query(QUERY_FEAT, IMAGE_NAME, query_dirty_threshold)
    #
    # f = open(testA_ranklist, encoding='utf-8')
    # content = f.read()
    # dic = json.loads(content)
    # cleaned_rank_dict_testA = {}
    # for query_name in list(dic.keys()):
    #     print(query_name)
    #     if query_name in clean_query_id_set:
    #         origin_ranklist = dic[query_name][:10]
    #         query_cur_feat = testA_query_feats[testA_query_imgnames.index(query_name)]
    #
    #         cleaned_ranklist = []
    #         cleaned_count = 0
    #         for gallery_name in origin_ranklist:
    #             rank_cur_feat = testA_gallery_feats[testA_gallery_imgnames.index(gallery_name)]
    #             score = np.dot(query_cur_feat, rank_cur_feat)
    #             if score > rank_dirty_threshold:
    #                 cleaned_ranklist.append(gallery_name)
    #                 cleaned_count += 1
    #             else:
    #                 break
    #         cleaned_rank_dict_testA[query_name] = cleaned_ranklist
    #         print(cleaned_count)
    #
    # f = open(testB_ranklist, encoding='utf-8')
    # content = f.read()
    # dic = json.loads(content)
    # cleaned_rank_dict_testB = {}
    # for query_name in list(dic.keys()):
    #     print(query_name)
    #     if query_name in clean_query_id_set:
    #         origin_ranklist = dic[query_name][:10]
    #         query_cur_feat = testB_query_feats[testB_query_imgnames.index(query_name)]
    #
    #         cleaned_ranklist = []
    #         cleaned_count = 0
    #         for gallery_name in origin_ranklist:
    #             rank_cur_feat = testB_gallery_feats[testB_gallery_imgnames.index(gallery_name)]
    #             score = np.dot(query_cur_feat, rank_cur_feat)
    #             if score > rank_dirty_threshold:
    #                 cleaned_ranklist.append(gallery_name)
    #                 cleaned_count += 1
    #             else:
    #                 break
    #         cleaned_rank_dict_testB[query_name] = cleaned_ranklist
    #         print(cleaned_count)
    #
    # count = 4768
    # for query, clean_list in tqdm(cleaned_rank_dict_testA.items()):
    #     input_path = os.path.join("/home/xiangan/data_reid/testA/query_a", query)
    #     output_name = os.path.join("/home/xiangan/code_and_data/train_split/all_final",
    #                                "%d_c1_%s" % (count, query))
    #     open(output_name, 'wb').write(open(input_path, 'rb').read())
    #     for clean_name in clean_list:
    #         input_path = os.path.join("/home/xiangan/data_reid/testA/gallery_a", clean_name)
    #         output_name = os.path.join("/home/xiangan/code_and_data/train_split/all_final",
    #                                    "%d_c1_%s" % (count, clean_name))
    #         open(output_name, 'wb').write(open(input_path, 'rb').read())
    #
    #     count += 1
    #
    # for query, clean_list in tqdm(cleaned_rank_dict_testB.items()):
    #     input_path = os.path.join("/home/xiangan/data_reid/testB/query_b", query)
    #     output_name = os.path.join("/home/xiangan/code_and_data/train_split/all_final",
    #                                "%d_c1_%s" % (count, query))
    #     open(output_name, 'wb').write(open(input_path, 'rb').read())
    #     for clean_name in clean_list:
    #         input_path = os.path.join("/home/xiangan/data_reid/testB/gallery_b", clean_name)
    #         output_name = os.path.join("/home/xiangan/code_and_data/train_split/all_final",
    #                                    "%d_c1_%s" % (count, clean_name))
    #         open(output_name, 'wb').write(open(input_path, 'rb').read())
    #
    #     count += 1


if __name__ == '__main__':
    main()
