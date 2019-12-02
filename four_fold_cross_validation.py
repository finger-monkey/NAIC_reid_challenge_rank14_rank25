import os

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
        else:
            LABEL_COUNT[_label] = 1

    def _split(lucky_num: int) -> tuple:
        """split origin image list to train, query, gallery sets
        """
        assert isinstance(lucky_num, int)
        _train_list = []
        _query_list = []
        _gallery_list = []
        _query_id_set = set()
        for _img_name in img_name_list:
            label = _img_name.split('_')[0]
            if int(label) % 4 == lucky_num:  # lucky id
                if label not in _query_id_set and LABEL_COUNT[label] != 1:
                    _query_list.append(_img_name)
                    _query_id_set.add(label)
                else:
                    _gallery_list.append(_img_name)
            else:
                _train_list.append(_img_name)
        return _train_list, _query_list, _gallery_list

    def _copy(source_root, target_root, image_list):
        for _img in image_list:
            source_path = os.path.join(source_root, _img)
            target_path = os.path.join(target_root, _img)
            open(target_path, 'wb').write(open(source_path, 'rb').read())

    for split_num in [0, 1, 2, 3]:
        train_list, query_list, gallery_list = _split(split_num)
        train_path = os.path.join(SPLIT_ROOT, "split%d/bounding_box_train" % (split_num + 1))
        query_path = os.path.join(SPLIT_ROOT, "split%d/query" % (split_num + 1))
        gallery_path = os.path.join(SPLIT_ROOT, "split%d/bounding_box_test" % (split_num + 1))

        os.makedirs(train_path)
        os.makedirs(query_path)
        os.makedirs(gallery_path)
        _copy(ALL_ROOT, train_path, train_list)
        _copy(ALL_ROOT, query_path, query_list)
        _copy(ALL_ROOT, gallery_path, gallery_list)


if __name__ == '__main__':
    main()
