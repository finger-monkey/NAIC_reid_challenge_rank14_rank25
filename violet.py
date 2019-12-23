import cv2
import os
import shutil
from numpy import mean

ROOT = 'D:\\train_split4_rematch\\train_split4_rematch\\split1\\query'
IMAGE_PATH = [
    '0010_c1_14345725.png',
    '0013_c1_42502329.png',
    '0015_c1_438428163.png',
    '0017_c1_760055344.png',
    # '00777536.png',
    # '00952768.png',
    # '00970383.png'
]


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


def main():
    [print(check(os.path.join(ROOT, x))) for x in os.listdir(ROOT)]


if __name__ == '__main__':
    main()
