import os
import pickle
import json
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

LABEL_DICT = {}

# threshold
RANK_DIRTY_THRESHOLD = 0.78
MERGE_THRESHOLD = 0.35

# features
GALLERY_FEAT = '/home/xiangan/dgreid/features/fighting_003_test_violet/gallery_feature.feat'
QUERY_FEAT = '/home/xiangan/dgreid/features/fighting_003_test_violet/query_feature.feat'

# output name
OUTPUT_NAME = "testB_violet_0.78_0.35"

bank_list = open('banklist').readlines()
bank_list = [x.strip() for x in bank_list]


def process_info(info):
    feats, img_names = info
    feats = preprocessing.normalize(feats)
    return feats, img_names


def main():
    #
    # testA
    testA_gallery_info = pickle.load(open(GALLERY_FEAT, 'rb'))
    testA_query_info = pickle.load(open(QUERY_FEAT, 'rb'))

    testA_gallery_feats, testA_gallery_img_names = process_info(testA_gallery_info)
    testA_query_feats, testA_query_img_names = process_info(testA_query_info)

    #
    sim = np.dot(testA_query_feats, testA_gallery_feats.T)
    num_q, num_g = sim.shape
    indices = np.argsort(-sim, axis=1)

    submission_key = {}
    for q_idx in range(num_q):
        order = indices[q_idx][:200]
        query_gallery = []
        for gallery_index in order:
            query_gallery.append(testA_gallery_img_names[gallery_index])
        submission_key[testA_query_img_names[q_idx]] = query_gallery

    #
    cls = AgglomerativeClustering(
        n_clusters=None,
        linkage='average',
        affinity="cosine",
        distance_threshold=MERGE_THRESHOLD
    )

    cls.fit(testA_query_feats)

    # generate label dict
    for idx, _label in enumerate(cls.labels_):
        if _label not in LABEL_DICT:
            LABEL_DICT[_label] = [testA_query_img_names[idx]]
        else:
            assert isinstance(LABEL_DICT[_label], list)
            LABEL_DICT[_label].append(testA_query_img_names[idx])

    print(len(set(cls.labels_)))

    cleaned_rank_dict_testA = {}

    dirty_count = 0
    dirty_query_set = set()

    cleaned_count = 0
    gallery2query_dict = {}
    for i in tqdm(LABEL_DICT.keys()):

        # query name
        cur_query_name = LABEL_DICT[i][0]

        # get query feat from testA_query_feats
        query_cur_feat = testA_query_feats[testA_query_img_names.index(cur_query_name)]

        #
        cleaned_rank_list = []
        origin_rank_list = submission_key[cur_query_name][:20]

        for gallery_name in origin_rank_list:
            try:
                rank_cur_feat = testA_gallery_feats[testA_gallery_img_names.index(gallery_name)]
            except:
                continue
            score = np.dot(query_cur_feat, rank_cur_feat)
            if score > RANK_DIRTY_THRESHOLD:
                if gallery_name in gallery2query_dict.keys():
                    print(gallery2query_dict[gallery_name])
                    print("%s vs %s" % (gallery2query_dict[gallery_name], cur_query_name))
                    dirty_count += 1
                    dirty_query_set.add(cur_query_name)
                else:
                    gallery2query_dict[gallery_name] = cur_query_name

                    if gallery_name in bank_list or cur_query_name in bank_list:
                        dirty_query_set.add(cur_query_name)

                #
                cleaned_rank_list.append(gallery_name)
                cleaned_count += 1
            else:
                break
        cleaned_rank_dict_testA[cur_query_name] = cleaned_rank_list
    print('cleaned:', cleaned_count)
    print('dirty_cout:', dirty_count)
    count = 30000

    output_path = "/data/anxiang/reid_extra/%s" % OUTPUT_NAME
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for query, clean_list in cleaned_rank_dict_testA.items():

        if query in dirty_query_set:
            continue

        input_path = os.path.join("/data/anxiang/reid/testB/violet/query", query)
        output_name = os.path.join("/data/anxiang/reid_extra/%s" % OUTPUT_NAME,
                                   "%d_c1_%s" % (count, query))
        open(output_name, 'wb').write(open(input_path, 'rb').read())
        for clean_name in clean_list:
            input_path = os.path.join("/data/anxiang/reid/testB/violet/gallery", clean_name)
            output_name = os.path.join("/data/anxiang/reid_extra/%s" % OUTPUT_NAME,
                                       "%d_c1_%s" % (count, clean_name))
            open(output_name, 'wb').write(open(input_path, 'rb').read())

        count += 1


if __name__ == '__main__':
    main()
