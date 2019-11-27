import json
import pickle

import numpy as np
import torch
from sklearn import preprocessing
from rerank.rerank_kreciprocal import re_ranking


def process_info(info):
    feats = []
    imgnames = []
    for i in range(len(info)):
        # print(info[i][0].flatten().shape)
        feats.append(info[i][0].flatten())
        imgnames.append(info[i][1])
    feats = np.array(feats)
    feats = preprocessing.normalize(feats)
    return feats, imgnames


def get(FEATURE, query_imgnames_1, gallery_imgnames_1):
    gallery_info_2 = pickle.load(open('%s/gallery_a_feature.feat' % FEATURE, 'rb'))
    query_info_2 = pickle.load(open('%s/query_a_feature.feat' % FEATURE, 'rb'))
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
    FEATURE_1 = "/home/xiangan/dgreid/features/402"
    FEATURE_2 = "/home/xiangan/dgreid/features/405"
    FEATURE_3 = "/home/xiangan/dgreid/features/114"
    # FEATURE_4 = "/home/xiangan/dgreid/features/113"
    # FEATURE_5 = "/home/xiangan/dgreid/features/056"
    # FEATURE_6 = "/home/xiangan/dgreid/features/111"
    # FEATURE_7 = "/home/xiangan/dgreid/features/feature_BFE_52"

    # feature_1
    gallery_info_1 = pickle.load(open('%s/gallery_a_feature.feat' % FEATURE_1, 'rb'))
    query_info_1 = pickle.load(open('%s/query_a_feature.feat' % FEATURE_1, 'rb'))
    gallery_feats_1, gallery_imgnames_1 = process_info(gallery_info_1)
    query_feats_1, query_imgnames_1 = process_info(query_info_1)

    qf_2, gf_2 = get(FEATURE_2, query_imgnames_1, gallery_imgnames_1)
    qf_3, gf_3 = get(FEATURE_3, query_imgnames_1, gallery_imgnames_1)
    # qf_4, gf_4 = get(FEATURE_4, query_imgnames_1, gallery_imgnames_1)
    # qf_5, gf_5 = get(FEATURE_5, query_imgnames_1, gallery_imgnames_1)
    # qf_6, gf_6 = get(FEATURE_6, query_imgnames_1, gallery_imgnames_1)
    # qf_7, gf_7 = get(FEATURE_7, query_imgnames_1, gallery_imgnames_1)

    query_feats = np.concatenate((query_feats_1, qf_2, qf_3), axis=1)
    gallery_feats = np.concatenate((gallery_feats_1, gf_2, gf_3), axis=1)
    query_feats = preprocessing.normalize(query_feats)
    gallery_feats = preprocessing.normalize(gallery_feats)

    query_feats = torch.from_numpy(query_feats)
    gallery_feats = torch.from_numpy(gallery_feats)
    sim = re_ranking(query_feats, gallery_feats, k1=7, k2=3, lambda_value=0.85)
    # rerank1 7 3 0.85
    # rerank2 7 3 0.8
    # rerank3 6 3 0.8

    # sim = np.dot(query_feats, gallery_feats.T)
    num_q, num_g = sim.shape
    # indices = np.argsort(-sim, axis=1)
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

    with open('11_27_rerank5.json', 'w', encoding='utf-8') as f:
        f.write(submission_json)


if __name__ == '__main__':
    main()
