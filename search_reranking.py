import pickle

import numpy as np
import torch
from ignite.metrics import Metric
from sklearn import preprocessing

from rerank.rerank_kreciprocal import re_ranking

SPLIT_NAME = 'split1'


def process_info(info):
    feats, imgnames = info
    feats = preprocessing.normalize(feats)
    return feats, imgnames


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=200):
    """
        Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class ReidMetric(Metric):
    def __init__(self, num_query, max_rank=200, feat_norm='yes'):
        super(ReidMetric, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.feats = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self, re_rank=None):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if re_rank is not None:
            #
            distmat = re_ranking(qf, gf, **re_rank)
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        else:
            #
            assert isinstance(gf, torch.Tensor)
            distmat = np.dot(qf, gf.T)
            distmat = -distmat
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        print("mAP: {:.1%}".format(mAP))
        for r in [1, 5]:
            print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return cmc, mAP


def main():
    gallery_info = pickle.load(open(
        '/home/xiangan/dgreid/features/dgreid_test_003/%s_gallery_feature.feat' % SPLIT_NAME, 'rb'))
    query_info = pickle.load(open(
        '/home/xiangan/dgreid/features/dgreid_test_003/%s_query_feature.feat' % SPLIT_NAME, 'rb'))

    gallery_feats, gallery_imgnames = process_info(gallery_info)
    query_feats, query_imgnames = process_info(query_info)

    sim = np.dot(query_feats, gallery_feats.T)
    num_q, num_g = sim.shape
    indices = np.argsort(-sim, axis=1)

    clean_set = set()

    #
    for q_idx in range(num_q):
        order = indices[q_idx][:70]
        for gallery_index in order:
            clean_set.add(gallery_imgnames[gallery_index])
        else:
            continue

    temp_feat = np.zeros((len(clean_set), gallery_feats.shape[1]))
    for idx, name in enumerate(clean_set):
        temp_feat[idx] = gallery_feats[gallery_imgnames.index(name)]

    print(temp_feat.shape)
    gallery_imgnames = list(clean_set)
    gallery_feats = temp_feat

    reid_metric = ReidMetric(num_query=len(query_imgnames))
    query_gallery_feat = torch.from_numpy(np.concatenate((query_feats, gallery_feats), axis=0))

    for index, image_name in enumerate(query_imgnames + gallery_imgnames):
        #
        pid = int(image_name.split('_')[0])
        camid = image_name.split('_')[1]
        reid_metric.update((torch.reshape(query_gallery_feat[index], (1, -1)), [pid], [camid]))

    reid_metric.compute()

    for k1 in [10]:
        for k2 in [2]:
            for l in [0.3]:
                print(k1, k2, l)
                kw = {
                    'k1': k1,
                    'k2': k2,
                    'lambda_value': l
                }
                reid_metric.compute(re_rank=kw)


if __name__ == '__main__':
    main()
