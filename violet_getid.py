import cv2
import os
import shutil
from numpy import mean
from tqdm import tqdm

ROOT = '/data/xiangan/reid_final/all/bounding_box_train'

#
DELETE_GALLERY_ROOT = "/data/xiangan/reid_final/violet_origin/bounding_box_test"
DELETE_QUERY_ROOT = "/data/xiangan/reid_final/violet_origin/query"

#
DELETE_TRAIN_ROOT = "/data/xiangan/reid_final/violet_del_train/bounding_box_train"


def info(image_path):
    """"""
    image_test = cv2.imread(image_path)
    b_mean = mean(image_test[:, :, 0])
    g_mean = mean(image_test[:, :, 1])
    r_mean = mean(image_test[:, :, 2])

    # print('b:%d\tg:%d\tr:%d\t' % (b_mean, g_mean, r_mean))
    return int(b_mean), int(g_mean), int(r_mean)


def check(image_path, threshold=300):
    return True if sum(info(image_path)) > threshold else False


def delete_img(delete_id_set, image_root):
    image_name_list = os.listdir(image_root)
    for image_name in tqdm(image_name_list):
        image_path = os.path.join(image_root, image_name)
        p_id = image_name.split('_')[0]
        if p_id in delete_id_set:
            os.unlink(image_path)


def main():
    violet_set = set()
    image_name_list = os.listdir(ROOT)

    for image_name in tqdm(image_name_list):

        # get image path
        image_path = os.path.join(ROOT, image_name)

        if check(image_path):
            p_id = image_name.split('_')[0]
            violet_set.add(p_id)
    for p_id in violet_set:
        print(p_id)
    delete_img(violet_set, DELETE_GALLERY_ROOT)
    delete_img(violet_set, DELETE_QUERY_ROOT)
    delete_img(violet_set, DELETE_TRAIN_ROOT)


if __name__ == '__main__':
    main()
