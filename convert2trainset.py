import os
import shutil

# config
ORIGIN_TRAIN_ROOT = "/tmp/data/train/image"
LABEL_FILE = "/tmp/data/train/label/train_list.txt"

TRAIN_ROOT = "/tmp/data/all/bounding_box_train"

_dirty_label = {
    '105180993.png': '5721',
    '829283568.png': '3180',
    '943445997.png': '3369'
}


def main():
    if not os.path.exists(TRAIN_ROOT):
        os.makedirs(TRAIN_ROOT)

    with open(LABEL_FILE, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            image_name = line.split(' ')[0].split('/')[1]
            if image_name in _dirty_label.keys():
                label = _dirty_label[image_name]
            else:
                label = line.split(' ')[1]

            label = int(label)
            target_name = "%4d_c1_%s" % (label, image_name)

            source_path = os.path.join(ORIGIN_TRAIN_ROOT, image_name)
            target_path = os.path.join(TRAIN_ROOT, target_name)
            shutil.copy(source_path, target_path)

    os.makedirs("/tmp/data/all/bounding_box_test")
    os.makedirs("/tmp/data/all/query")


if __name__ == '__main__':
    main()
