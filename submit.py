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


gallery_info = pickle.load(open('/home/xiangan/data_match/feat_11_1_300_300/gallery_a_feature.feat', 'rb'))
query_info = pickle.load(open('/home/xiangan/data_match/feat_11_1_300_300/query_a_feature.feat', 'rb'))

gallery_feats, gallery_imgnames = process_info(gallery_info)
query_feats, query_imgnames = process_info(query_info)

#
query_feats = torch.from_numpy(query_feats)
gallery_feats = torch.from_numpy(gallery_feats)
sim = re_ranking(query_feats, gallery_feats, k1=7, k2=3, lambda_value=0.85)


# sim = np.dot(query_feats, gallery_feats.T)
num_q, num_g = sim.shape
# indices = np.argsort(-sim, axis=1)
indices = np.argsort(sim, axis=1)

submission_key = {}
for q_idx in range(num_q):
    order = indices[q_idx][:200]
    query_gallery = []
    for gallery_index in order:
        query_gallery.append(gallery_imgnames[gallery_index])
    submission_key[query_imgnames[q_idx]] = query_gallery

submission_json = json.dumps(submission_key)
print(type(submission_json))

with open('rerank_11_1.json', 'w', encoding='utf-8') as f:
    f.write(submission_json)
