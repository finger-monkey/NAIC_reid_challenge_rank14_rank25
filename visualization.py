import json
import os
import pickle

import numpy as np
from sklearn import preprocessing
from tqdm import tqdm


def visualization(rank_dict, test_root, output_root):
    query_root = os.path.join(test_root, 'query_a')
    gallery_root = os.path.join(test_root, 'gallery_a')
    for query in rank_dict.keys():
        query_folder = os.path.join(output_root, "%d_%s" % (len(rank_dict[query]), query))
        os.makedirs(query_folder)
        open(os.path.join(query_folder, query), 'wb').write(open(os.path.join(query_root, query), 'rb').read())
        for ranid, neighbor in enumerate(rank_dict[query]):
            target_path = os.path.join(query_folder, "%d_%s" % (ranid + 1, neighbor))
            source_path = os.path.join(gallery_root, neighbor)
            open(target_path, 'wb').write(open(source_path, 'rb').read())


def main():
    f = open('ensemblex7.json', encoding='utf-8')

    #
    content = f.read()
    dic = json.loads(content)

    #
    for query_name in list(dic.keys()):
        print(query_name)
        origin_ranklist = dic[query_name][:10]
        for gallery_name in origin_ranklist:
            rank_cur_feat = gallery_feats[gallery_imgnames.index(gallery_name)]
            score = np.dot(query_cur_feat, rank_cur_feat)
            if score > rank_dirty_threshold:
                cleaned_ranklist.append(gallery_name)
                cleaned_count += 1
            else:
                break
        cleaned_rank_dict[query_name] = cleaned_ranklist
        # print(cleaned_count)

    # visualization(
    #     cleaned_rank_dict,
    #     '/home/xiangan/data_reid/testA',
    #     '/home/xiangan/data_reid/visualization/11_21')
    # a = open('ensemblex7.json')
    # print(a.readlines()[0])
    # f = open('ensemblex7.json', encoding='utf-8')
    # content = f.read()
    # dic = json.loads(content)
    # for i in dic.keys():
    #     print(i)
    #     print(dic[i])
    # print(dic[dic.keys()])

    count = 4768
    for query, clean_list in tqdm(cleaned_rank_dict.items()):
        input_path = os.path.join("/home/xiangan/data_reid/testA/query_a", query)
        output_name = os.path.join("/home/xiangan/code_and_data/train_split/test_extra",
                                   "%d_c1_%s" % (count, query))
        open(output_name, 'wb').write(open(input_path, 'rb').read())
        for clean_name in clean_list:
            input_path = os.path.join("/home/xiangan/data_reid/testA/gallery_a", clean_name)
            output_name = os.path.join("/home/xiangan/code_and_data/train_split/test_extra",
                                       "%d_c1_%s" % (count, clean_name))
            open(output_name, 'wb').write(open(input_path, 'rb').read())

        count += 1


if __name__ == '__main__':
    main()
