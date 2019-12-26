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
RANK_DIRTY_THRESHOLD = 0.85
MERGE_THRESHOLD = 0.4

# features
GALLERY_FEAT = '/home/xiangan/dgreid/features/apex_003_test_origin/gallery_feature.feat'
QUERY_FEAT = '/home/xiangan/dgreid/features/apex_003_test_origin/query_feature.feat'

# rank list
RANK_LIST = 'ensemble_xxx.json'

# output name
OUTPUT_NAME = "test1"


def process_info(info):
    feats, img_names = info
    feats = preprocessing.normalize(feats)
    return feats, img_names


def main():
    #
    # testA rank_list
    testA_rank_list = RANK_LIST

    # testA
    testA_gallery_info = pickle.load(open(GALLERY_FEAT, 'rb'))
    testA_query_info = pickle.load(open(QUERY_FEAT, 'rb'))

    testA_gallery_feats, testA_gallery_img_names = process_info(testA_gallery_info)
    testA_query_feats, testA_query_img_names = process_info(testA_query_info)

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

    f = open(testA_rank_list, encoding='utf-8')
    content = f.read()
    dic = json.loads(content)

    cleaned_rank_dict_testA = {}

    clean_set = set()
    dirty_count = 0
    cleaned_count = 0
    for i in tqdm(LABEL_DICT.keys()):

        # query name
        cur_query_name = LABEL_DICT[i][0]

        # get query feat from testA_query_feats
        query_cur_feat = testA_query_feats[testA_query_img_names.index(cur_query_name)]

        #
        cleaned_rank_list = []
        origin_rank_list = dic[cur_query_name][:100]

        gallery2query_dict = {}


        for gallery_name in origin_rank_list:
            try:
                rank_cur_feat = testA_gallery_feats[testA_gallery_img_names.index(gallery_name)]
            except:
                continue
            score = np.dot(query_cur_feat, rank_cur_feat)
            if score > RANK_DIRTY_THRESHOLD:
                if gallery_name in gallery2query_dict:
                    print(gallery2query_dict[gallery_name])
                    print("%s vs %s" % (gallery2query_dict[gallery_name], cur_query_name))
                    dirty_count += 1
                else:
                    gallery2query_dict[gallery_name] = cur_query_name

                #
                cleaned_rank_list.append(gallery_name)
                cleaned_count += 1
            else:
                break
        cleaned_rank_dict_testA[cur_query_name] = cleaned_rank_list
    print('cleaned:', cleaned_count)
    print('dirty_cout:', dirty_count)
    count = 15000

    output_path = "/data/xiangan/reid_extra/%s" % OUTPUT_NAME
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for query, clean_list in cleaned_rank_dict_testA.items():
        input_path = os.path.join("/data/xiangan/reid_final/test/query_a", query)
        output_name = os.path.join("/data/xiangan/reid_extra/%s" % OUTPUT_NAME,
                                   "%d_c1_%s" % (count, query))
        open(output_name, 'wb').write(open(input_path, 'rb').read())
        for clean_name in clean_list:
            input_path = os.path.join("/data/xiangan/reid_final/test/gallery_a", clean_name)
            output_name = os.path.join("/data/xiangan/reid_extra/%s" % OUTPUT_NAME,
                                       "%d_c1_%s" % (count, clean_name))
            open(output_name, 'wb').write(open(input_path, 'rb').read())

        count += 1


if __name__ == '__main__':
    main()
