from utils.reid_metric import R1_mAP
import numpy as np
from sklearn import preprocessing
import pickle


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


gallery_info = pickle.load(open('/home/xiangan/dgreid/exps/test/query_train_feature.feat', 'rb'))
query_info = pickle.load(open('/home/xiangan/dgreid/exps/test/gallery_train_feature.feat', 'rb'))

gallery_feats, gallery_imgnames = process_info(gallery_info)
query_feats, query_imgnames = process_info(query_info)

a = R1_mAP(550, 200, 'yes')
