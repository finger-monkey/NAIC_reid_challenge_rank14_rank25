import json
import pickle

import numpy as np
import torch
from sklearn import preprocessing
from rerank.rerank_kreciprocal import re_ranking

NAME = "apex_003"
VIOLET = "final_001_test_violet"


def process_info(info):
    feats, imgnames = info
    feats = preprocessing.normalize(feats)
    return feats, imgnames


def trans(source_image_name, target_image_name: list,
          source_feats, target_feats):
    for image_name in source_image_name:
        index = target_image_name.index(image_name)
        target_feats[index] = source_feats[index]

    return target_feats


gallery_info = pickle.load(open('features/%s/gallery_feature.feat' % NAME, 'rb'))
query_info = pickle.load(open('features/%s/query_feature.feat' % NAME, 'rb'))

gallery_feats, gallery_imgnames = process_info(gallery_info)
query_feats, query_imgnames = process_info(query_info)
#
violet_gallery_info = pickle.load(open('features/%s/gallery_feature.feat' % VIOLET, 'rb'))
violet_query_info = pickle.load(open('features/%s/query_feature.feat' % VIOLET, 'rb'))

violet_gallery_feats, violet_gallery_imgnames = process_info(gallery_info)
violet_query_feats, violet_query_imgnames = process_info(query_info)

#
query_feats = trans(source_image_name=violet_query_imgnames,
                    source_feats=violet_query_feats,
                    target_image_name=query_imgnames,
                    target_feats=query_feats)

gallery_feats = trans(source_image_name=violet_gallery_imgnames,
                      source_feats=violet_gallery_feats,
                      target_image_name=gallery_imgnames,
                      target_feats=gallery_feats)

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

with open('rerank_%s.json' % NAME, 'w', encoding='utf-8') as f:
    f.write(submission_json)
