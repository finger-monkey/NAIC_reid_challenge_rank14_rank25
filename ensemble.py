import pickle

import numpy as np
from sklearn import preprocessing

FEATURE_LIST = [
    "dgreid_001", "dgreid_002", "dgreid_b_003", "dgreid_b_004"
]


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

    pickle.dump([query_feats, query_imgnames_1],
                open('/home/xiangan/dgreid/features/ensemble/query_feature.feat', 'wb'))
    pickle.dump([gallery_feats, gallery_imgnames_1],
                open('/home/xiangan/dgreid/features/ensemble/gallery_feature.feat', 'wb'))


if __name__ == '__main__':
    main()
