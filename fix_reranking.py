import numpy as np
from sklearn import preprocessing
import pickle
import re
import torch
from ignite.metrics import Metric
from data.datasets.eval_reid import eval_func
from utils.re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self, rerank_func=None, **kwargs):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            # print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        if rerank_func is not None:
            distmat = rerank_func(qf, gf, kwargs['k1'], kwargs['k2'], kwargs['l'])
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


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


gallery_info = pickle.load(open('/home/xiangan/dgreid/exps/test/gallery_train_feature.feat', 'rb'))
query_info = pickle.load(open('/home/xiangan/dgreid/exps/test/query_train_feature.feat', 'rb'))

gallery_feats, gallery_imgnames = process_info(gallery_info)
query_feats, query_imgnames = process_info(query_info)

query_feats = torch.from_numpy(query_feats)
gallery_feats = torch.from_numpy(gallery_feats)

# print(query_feats.shape)
# print(gallery_feats.shape)


a = R1_mAP(550, 200, 'yes')

pattern = re.compile(r'([-\d]+)_c(\d)')
for i in range(len(query_imgnames)):
    img_path = query_imgnames[i]
    feat = query_feats[i].unsqueeze(0)
    pid, camid = map(int, pattern.search(img_path).groups())
    a.update((feat, [pid], [camid]))

for i in range(len(gallery_imgnames)):
    img_path = gallery_imgnames[i]
    feat = gallery_feats[i].unsqueeze(0)
    pid, camid = map(int, pattern.search(img_path).groups())
    a.update((feat, [pid], [camid]))

cmc, map = a.compute()
for r in [1]:
    print("CMC curve, Rank-%d:%.4f, map:%.4f" % (r, cmc[r - 1], map))

for k1 in range(5, 10, 2):
    for k2 in range(5, 11, 2):
        for l in [0.5, 0.7, 0.9]:
            print("====k1=%d=====k2=%d=====l=%f" % (k1, k2, l))
            cmc, map = a.compute(re_ranking, k1=k1, k2=k2, l=l)
            for r in [1]:
                print("CMC curve, Rank-%d:%.4f, map:%.4f" % (r, cmc[r - 1], map))
