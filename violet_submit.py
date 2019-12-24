import json
import pickle

import numpy as np
import torch
from sklearn import preprocessing
from rerank.rerank_kreciprocal import re_ranking

ORIGIN = "apex_003_test_origin"
VIOLET = "violet_003_test_another"


def process_info(info):
    feats, imgnames = info
    feats = preprocessing.normalize(feats)
    return feats, imgnames


gallery_info_origin = pickle.load(open('features/%s/gallery_feature.feat' % ORIGIN, 'rb'))
query_info_origin = pickle.load(open('features/%s/query_feature.feat' % ORIGIN, 'rb'))

gallery_feats_origin, gallery_imgnames_origin = process_info(gallery_info_origin)
query_feats_origin, query_imgnames_origin = process_info(query_info_origin)

gallery_info_violet = pickle.load(open('features/%s/gallery_feature.feat' % VIOLET, 'rb'))
query_info_violet = pickle.load(open('features/%s/query_feature.feat' % VIOLET, 'rb'))

gallery_feats_violet, gallery_imgnames_violet = process_info(gallery_info_violet)
query_feats_violet, query_imgnames_violet = process_info(query_info_violet)

# concat
query_feats = np.concatenate([query_feats_origin, query_feats_violet])
gallery_feats = np.concatenate([gallery_feats_origin, gallery_feats_violet])

query_imgnames = query_imgnames_origin + query_imgnames_violet
gallery_imgnames = gallery_imgnames_origin + gallery_imgnames_violet

#
query_feats = torch.from_numpy(query_feats)
gallery_feats = torch.from_numpy(gallery_feats)
sim = re_ranking(query_feats, gallery_feats, k1=7, k2=3, lambda_value=0.80)

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

with open('rerank_violet_%s.json', 'w', encoding='utf-8') as f:
    f.write(submission_json)
