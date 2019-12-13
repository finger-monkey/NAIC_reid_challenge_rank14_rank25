import os
import shutil

PATH_SOURCE_QUERY = "/data/xiangan/reid_final/test/query_a"
PATH_SOURCE_GALLERY = "/data/xiangan/reid_final/test/gallery_a"

PATH_TARGET = "/data/xiangan/reid_final/uncorrelated"


def main():

    def c(source_root, target_root):
        image_list = os.path.join(source_root)
        for img_path in image_list:
            source_path = os.path.join(source_root, img_path)
            target_path = os.path.join(target_root, "9999_c2_%s" % img_path[6:])
            shutil.copy(source_path, target_path)

    c(PATH_SOURCE_QUERY, PATH_TARGET)
    c(PATH_SOURCE_GALLERY, PATH_TARGET)


if __name__ == '__main__':
    main()
