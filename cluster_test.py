import os
import pickle
import json
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering

LABEL_DICT = {}
# threshold
rank_dirty_threshold = 0.67
merge_threshold = 0.45


def process_info(info):
    feats, img_names = info
    feats = preprocessing.normalize(feats)
    return feats, img_names


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


def visualization(rank_dict, test_root, output_root):
    for i in rank_dict.keys():
        each_folder = os.path.join(output_root, "%d_%s" % (len(rank_dict[i]), str(i)))
        os.makedirs(each_folder)
        for ranid, neighbor in enumerate(rank_dict[i]):
            target_path = os.path.join(each_folder, "%d_%s" % (ranid + 1, neighbor))
            source_path = os.path.join(test_root, neighbor)
            open(target_path, 'wb').write(open(source_path, 'rb').read())


def main():
    #
    # testA rank_list
    testA_ranklist = 'submission_jiankang_train_001.json'

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
        distance_threshold=merge_threshold
    )

    cls.fit(testA_query_feats)

    # generate label dict
    for idx, _label in enumerate(cls.labels_):
        if _label not in LABEL_DICT:
            LABEL_DICT[_label] = [testA_query_img_names[idx]]
        else:
            assert isinstance(LABEL_DICT[_label], list)
            LABEL_DICT[_label].append(testA_query_img_names[idx])

    # visualization(
    #     LABEL_DICT,
    #     "/data/xiangan/reid_final/test/query_a",
    #     "/data/xiangan/vis_query"
    # )
    print(len(set(cls.labels_)))

    f = open(testA_ranklist, encoding='utf-8')
    content = f.read()
    dic = json.loads(content)
    cleaned_rank_dict_testA = {}
    gallery_set = set()
    cleaned_count = 0
    for i in LABEL_DICT.keys():
        cur_query_name = LABEL_DICT[i][0]

        query_cur_feat = testA_query_feats[testA_query_img_names.index(cur_query_name)]
        cleaned_ranklist = []
        origin_ranklist = dic[cur_query_name][:30]

        for gallery_name in origin_ranklist:
            rank_cur_feat = testA_gallery_feats[testA_gallery_img_names.index(gallery_name)]
            score = np.dot(query_cur_feat, rank_cur_feat)
            if score > rank_dirty_threshold:
                #
                if gallery_name in gallery_set:
                    continue
                else:
                    gallery_set.add(gallery_name)

                #
                cleaned_ranklist.append(gallery_name)
                cleaned_count += 1
            else:
                break
        cleaned_rank_dict_testA[cur_query_name] = cleaned_ranklist
    print(cleaned_count)

    # visualization_a(
    #     cleaned_rank_dict_testA,
    #     "/data/xiangan/reid_final/test/",
    #     "/data/xiangan/vis_query"
    # )
    count = 10000
    for query, clean_list in tqdm(cleaned_rank_dict_testA.items()):
        input_path = os.path.join("/data/xiangan/reid_final/test/query_a", query)
        output_name = os.path.join("/data/xiangan/reid_final/extra_3",
                                   "%d_c1_%s" % (count, query))
        open(output_name, 'wb').write(open(input_path, 'rb').read())
        for clean_name in clean_list:
            input_path = os.path.join("/data/xiangan/reid_final/test/gallery_a", clean_name)
            output_name = os.path.join("/data/xiangan/reid_final/extra_3",
                                       "%d_c1_%s" % (count, clean_name))
            open(output_name, 'wb').write(open(input_path, 'rb').read())

        count += 1



    # for i in list(dic.keys()):
    #     origin_ranklist = dic[i][:10]
    #     query_cur_feat = testA_query_feats[testA_query_img_names.index(query_name)]
    #
    #     cleaned_ranklist = []
    #     cleaned_count = 0
    #     for gallery_name in origin_ranklist:
    #         rank_cur_feat = testA_gallery_feats[testA_gallery_imgnames.index(gallery_name)]
    #         score = np.dot(query_cur_feat, rank_cur_feat)
    #         if score > rank_dirty_threshold:
    #             cleaned_ranklist.append(gallery_name)
    #             cleaned_count += 1
    #         else:
    #             break
    #     cleaned_rank_dict_testA[query_name] = cleaned_ranklist
    #     print(cleaned_count)
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
