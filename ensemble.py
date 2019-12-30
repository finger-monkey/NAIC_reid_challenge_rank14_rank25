import json
import pickle

import numpy as np
import torch
from sklearn import preprocessing

from rerank.rerank_kreciprocal import re_ranking

FEATURE_LIST = [
    "fighting_003", "fighting_002"
]

ENSEMBLE_NAME = "fighting_002_003"


def process_info(info):
    feats, imgnames = info
    feats = preprocessing.normalize(feats)
    return feats, imgnames


def get(FEATURE, query_imgnames_1, gallery_imgnames_1):
    gallery_info_2 = pickle.load(open('features/%s/gallery_feature.feat' % FEATURE, 'rb'))
    query_info_2 = pickle.load(open('features/%s/query_feature.feat' % FEATURE, 'rb'))
    gallery_feats_2, gallery_imgnames_2 = process_info(gallery_info_2)
    query_feats_2, query_imgnames_2 = process_info(query_info_2)
    #
    qf_2 = np.zeros_like(query_feats_2)
    gf_2 = np.zeros_like(gallery_feats_2)
    for i in range(len(query_imgnames_1)):
        index = query_imgnames_2.index(query_imgnames_1[i])
        qf_2[i] = query_feats_2[index]

    for i in range(len(gallery_imgnames_1)):
        index = gallery_imgnames_2.index(gallery_imgnames_1[i])
        gf_2[i] = gallery_feats_2[index]

    return qf_2, gf_2


def main():
    assert len(FEATURE_LIST) > 1

    # feature_1
    FEATURE_1 = FEATURE_LIST[0]

    gallery_info_1 = pickle.load(open('features/%s/gallery_feature.feat' % FEATURE_1, 'rb'))
    query_info_1 = pickle.load(open('features/%s/query_feature.feat' % FEATURE_1, 'rb'))
    gallery_feats_1, gallery_imgnames_1 = process_info(gallery_info_1)
    query_feats_1, query_imgnames_1 = process_info(query_info_1)

    # concat feat
    concat_query_list = [query_feats_1]
    concat_gallery_list = [gallery_feats_1]
    for _FEATURE in FEATURE_LIST[1:]:
        _query_feat, _gallery_feat = get(_FEATURE, query_imgnames_1, gallery_imgnames_1)
        concat_query_list.append(_query_feat)
        concat_gallery_list.append(_gallery_feat)

    query_feats = np.concatenate(concat_query_list, axis=1)
    gallery_feats = np.concatenate(concat_gallery_list, axis=1)

    query_feats = preprocessing.normalize(query_feats)
    gallery_feats = preprocessing.normalize(gallery_feats)

    query_feats = torch.from_numpy(query_feats)
    gallery_feats = torch.from_numpy(gallery_feats)
    sim = re_ranking(query_feats, gallery_feats, k1=7, k2=3, lambda_value=0.80)
    # rerank1 7 3 0.85
    # rerank2 7 3 0.8
    # rerank3 6 3 0.8

    num_q, num_g = sim.shape
    indices = np.argsort(sim, axis=1)

    submission_key = {}
    for q_idx in range(num_q):
        order = indices[q_idx][:200]
        query_gallery = []
        for gallery_index in order:
            query_gallery.append(gallery_imgnames_1[gallery_index])
        submission_key[query_imgnames_1[q_idx]] = query_gallery

    submission_json = json.dumps(submission_key)
    print(type(submission_json))

    with open('ensemble_%s.json' % ENSEMBLE_NAME, 'w', encoding='utf-8') as f:
        f.write(submission_json)


if __name__ == '__main__':
    main()
