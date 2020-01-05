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

    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)


    origin_query_path = "/tmp/data/origin/query"
    origin_gallery_path = "/tmp/data/origin/bounding_box_test"

    violet_query_path = "/tmp/data/violet/query"
    violet_gallery_path = '/tmp/data/violet/bounding_box_test'

    query_B = "/tmp/data/test/query_B"
    gallery_B = "/tmp/data/test/gallery_B"

    mkdir(origin_query_path)
    mkdir(origin_gallery_path)
    mkdir(violet_query_path)
    mkdir(violet_gallery_path)

    # query
    copy(
        source_root=query_B,
        origin_target_root=origin_query_path,
        violet_target_root=violet_query_path,
    )
    copy(
        source_root=gallery_B,
        origin_target_root=origin_gallery_path,
        violet_target_root=violet_gallery_path,
    )
