import pickle
import argparse
import numpy as np
from sklearn import preprocessing
import json


def process_info(info):
    feats = []
    imgnames = []
    for i in range(len(info)):
        feats.append(info[i][0].flatten())
        imgnames.append(info[i][1])
    feats = np.array(feats)
    feats = preprocessing.normalize(feats)
    return feats, imgnames


gallery_info = pickle.load(open('/home/xiangan/dgreid/features/024', 'rb'))
query_info = pickle.load(open('/home/xiangan/dgreid/features/024', 'rb'))

gallery_feats, gallery_imgnames = process_info(gallery_info)
query_feats, query_imgnames = process_info(query_info)
