import pickle
import numpy as np
from sklearn import preprocessing
import json


def cal_topn(q_feats, q_ids, d_feats, d_ids, max_rank=200):
    sim = np.dot(q_feats, d_feats.T)
    num_q, num_d = sim.shape
    print('num_q', num_q, 'num_d', num_d)
    indices = np.argsort(-sim, axis=1)
    matches = (d_ids[indices] == q_ids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    num_valid_q = 0.
    all_AP = []
    for q_idx in range(num_q):
        q_id = q_ids[q_idx]
        order = indices[q_idx]
        orig_cmc = matches[q_idx]
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    mAP = np.mean(all_AP)
    print('mAP', mAP)
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    print('top1', all_cmc[0])
    print('top5', all_cmc[4])
    print('top10', all_cmc[9])


def process_info(info):
    feats = []
    imgnames = []
    ids = []
    for i in range(len(info)):
        # print(info[i][0].flatten().shape)
        feats.append(info[i][0].flatten())
        imgnames.append(info[i][1])
        ids.append(info[i][1][:4])
    feats = np.array(feats)
    feats = preprocessing.normalize(feats)
    ids = np.array(ids)
    return feats, imgnames, ids


gallery_info = pickle.load(open('exps/test/gallery_train_feature.feat', 'rb'))
query_info = pickle.load(open('exps/test/query_train_feature.feat', 'rb'))

gallery_feats, gallery_imgnames, gallery_ids = process_info(gallery_info)
query_feats, query_imgnames, query_ids = process_info(query_info)
cal_topn(query_feats, query_ids, gallery_feats, gallery_ids, max_rank=100)
sim = np.dot(query_feats, gallery_feats.T)
num_q, num_g = sim.shape
indices = np.argsort(-sim, axis=1)

submission_key = {}
for q_idx in range(num_q):
    order = indices[q_idx][:200]
    query_gallery = []
    for gallery_index in order:
        query_gallery.append(gallery_imgnames[gallery_index])
    submission_key[query_imgnames[q_idx]] = query_gallery

submission_json = json.dumps(submission_key)
print(type(submission_json))

with open('submission_example_train.json', 'w', encoding='utf-8') as f:
    f.write(submission_json)
