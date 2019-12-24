import cv2
import os
import shutil
from numpy import mean
from tqdm import tqdm


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


def delete_img(move_id_set, image_root, another_root):
    image_name_list = os.listdir(image_root)
    for image_name in tqdm(image_name_list):
        source_image_path = os.path.join(image_root, image_name)
        target_image_path = os.path.join(another_root, image_name)
        p_id = image_name.split('_')[0]
        if p_id in move_id_set:
            shutil.move(source_image_path, target_image_path)


def copy(source_root, origin_target_root, violet_target_root):
    """"""
    image_name_list = os.listdir(source_root)

    for image_name in tqdm(image_name_list):

        # get image path
        source_image_path = os.path.join(source_root, image_name)
        if check(source_image_path):
            target_image_path = os.path.join(violet_target_root, image_name)
            shutil.copy(source_image_path, target_image_path)
        else:
            target_image_path = os.path.join(origin_target_root, image_name)
            shutil.copy(source_image_path, target_image_path)


if __name__ == '__main__':
    copy(
        source_root="/data/xiangan/reid_final/test/query_a",
        origin_target_root="/data/xiangan/reid_final/test_origin/query_a",
        violet_target_root="/data/xiangan/reid_final/test_violet/query_a",
    )
    copy(
        source_root="/data/xiangan/reid_final/test/gallery_a",
        origin_target_root="/data/xiangan/reid_final/test_origin/gallery_a",
        violet_target_root="/data/xiangan/reid_final/test_violet/gallery_a",
    )
