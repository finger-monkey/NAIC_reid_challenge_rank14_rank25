import os
import random
from sklearn.model_selection import KFold

# all dataset root
ALL_ROOT = "/train/trainset/1/datasets/trainset/bounding_box_train"
SPLIT_ROOT = "/train/trainset/1/datasets/"
LABEL_COUNT = {}


# 1280_c1_700108186.png
def main():
    # get label_count
    img_name_list = os.listdir(ALL_ROOT)
    for img_name in img_name_list:
        _label = img_name.split('_')[0]
        if _label in LABEL_COUNT:
            LABEL_COUNT[_label] += 1

    # shuffle image_list
    random.shuffle(img_name_list)

    # label list
    label_list = list(LABEL_COUNT.keys())

    def _split(train_index, test_index) -> tuple:
        """split origin image list to train, query, gallery sets
        """
        _train_id_list = [label_list[_i] for _i in train_index]
        _test_id_list = [label_list[_i] for _i in test_index]

        _train_list = []
        _query_list = []
        _gallery_list = []
        _query_id_set = set()
        for _img_name in img_name_list:

            # get label
            label = _img_name.split('_')[0]
            if label in _test_id_list:  # lucky id
                if label not in _query_id_set and LABEL_COUNT[label] != 1:
                    _query_list.append(_img_name)
                    _query_id_set.add(label)
                else:
                    _gallery_list.append(_img_name)
            elif label in _train_id_list:
                _train_list.append(_img_name)
            else:
                raise ValueError
        return _train_list, _query_list, _gallery_list

    def _copy(source_root, target_root, image_list):
        for _img in image_list:
            source_path = os.path.join(source_root, _img)
            target_path = os.path.join(target_root, _img)
            open(target_path, 'wb').write(open(source_path, 'rb').read())

    kfold = KFold(n_splits=4, shuffle=True, random_state=10086)
    for idx, (train_index, test_index) in enumerate(kfold.split(label_list)):
        train_list, query_list, gallery_list = _split(train_index, test_index)
        train_path = os.path.join(SPLIT_ROOT, "split%d/bounding_box_train" % (idx + 1))
        query_path = os.path.join(SPLIT_ROOT, "split%d/query" % (idx + 1))
        gallery_path = os.path.join(SPLIT_ROOT, "split%d/bounding_box_test" % (idx + 1))

        os.makedirs(train_path)
        os.makedirs(query_path)
        os.makedirs(gallery_path)
        _copy(ALL_ROOT, train_path, train_list)
        _copy(ALL_ROOT, query_path, query_list)
        _copy(ALL_ROOT, gallery_path, gallery_list)


if __name__ == '__main__':
    main()
