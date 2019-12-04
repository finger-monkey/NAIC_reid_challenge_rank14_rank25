import pickle
import random
import shutil


def process(train_key, test_key, path, train_map):
    for i in train_key:
        for img_path in train_map[i]:
            shutil.copy(img_path, path + 'bounding_box_train/' + i.zfill(4) + '_c1_' + img_path[6:])
    for i in test_key:
        if len(train_map[i]) < 2:
            shutil.copy(train_map[i][0], path + 'bounding_box_test/' + i.zfill(4) + '_c2_' + train_map[i][0][6:])
            continue
            # print(i ,train_map[i])
        shutil.copy(train_map[i][0], path + 'query/' + i.zfill(4) + '_c1_' + train_map[i][0][6:])
        for img_path in train_map[i][1:]:
            shutil.copy(img_path, path + 'bounding_box_test/' + i.zfill(4) + '_c2_' + img_path[6:])


train_list = open('train_list.txt', 'r').readlines()

train_map = {}
for img_info in train_list:
    img_path, img_id = img_info.strip().split(' ')
    if img_id not in train_map.keys():
        train_map[img_id] = [img_path]
    else:
        train_map[img_id].append(img_path)

train_key = list(train_map.keys())
random.shuffle(train_key)
pickle.dump(train_key, open('train_key', 'wb'))
num_train_key = len(train_key)
num_split = int(num_train_key / 5)
train_key1 = train_key[:num_split]
train_key2 = train_key[num_split:2 * num_split]
train_key3 = train_key[2 * num_split:3 * num_split]
train_key4 = train_key[3 * num_split:4 * num_split]
train_key5 = train_key[4 * num_split:]

path1 = '../../train_split/split1/'
path2 = '../../train_split/split2/'
path3 = '../../train_split/split3/'
path4 = '../../train_split/split4/'
path5 = '../../train_split/split5/'

process(train_key1 + train_key2 + train_key3 + train_key4, train_key5, path5, train_map)

process(train_key1 + train_key2 + train_key3 + train_key5, train_key4, path4, train_map)

process(train_key1 + train_key2 + train_key5 + train_key4, train_key3, path3, train_map)
process(train_key1 + train_key5 + train_key3 + train_key4, train_key2, path2, train_map)
process(train_key5 + train_key2 + train_key3 + train_key4, train_key1, path1, train_map)
