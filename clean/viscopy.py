import os
from tqdm import tqdm


class Copy:
    def __init__(self, source_root, target_root):
        self.source_root = source_root
        self.target_root = target_root

    def __call__(self):
        #
        path_list = self.get_path_list()
        print(path_list)
        #
        for idx, path in tqdm(enumerate(path_list)):
            # absolute path
            print(self.source_root)
            print(path)
            source_path = os.path.join(self.source_root, path)
            target_path = os.path.join(self.target_root, path)
            # copy
            open(target_path, 'wb').write(open(source_path, 'rb').read())

    def get_path_list(self) -> list:
        raise NotImplementedError


class CopyDirtyImage(Copy):
    def __init__(self, source_root, target_root, dirty_file):
        super().__init__(source_root, target_root)
        self.dirty_file = dirty_file

    def get_path_list(self) -> list:
        return [x.strip() for x in open(self.dirty_file).readlines()]


if __name__ == '__main__':

    #
    source_root = '/home/xiangan/code_and_data/train_split/all_extra/bounding_box_train',
    target_root = '/home/xiangan/code_and_data/repeat_dirty'
    dirty_file_root = '/home/xiangan/dgreid/clean/dirty_0.99'

    dirty_file_list = os.listdir(dirty_file_root)

    for dirty_file in dirty_file_list:
        target = os.path.join(target_root, dirty_file)
        if not os.path.exists(target):
            os.makedirs(target)
        CopyDirtyImage(
            source_root=source_root,
            target_root=str(target),
            dirty_file=str(os.path.join(dirty_file_root, dirty_file))
        )()
