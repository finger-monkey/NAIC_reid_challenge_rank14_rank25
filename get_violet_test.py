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
        source_root="/data/anxiang/reid/testB/query_b",
        origin_target_root="/data/anxiang/reid/testB/origin/query",
        violet_target_root="/data/anxiang/reid/testB/violet/query",
    )
    copy(
        source_root="/data/anxiang/reid/testB/gallery_b",
        origin_target_root="/data/anxiang/reid/testB/origin/gallery",
        violet_target_root="/data/anxiang/reid/testB/violet/gallery",
    )
