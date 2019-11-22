import argparse
import os
import pickle
import sys

import torch
from PIL import Image
from torch.backends import cudnn
from torchvision import transforms

sys.path.append('.')
from config import cfg
from modeling import build_model
from utils.logger import setup_logger
from tqdm import tqdm


class Feature:
    """
    get feature vector
    """

    def __init__(self, model, mean, std, size):
        self.model = model
        self.mean = mean
        self.std = std
        self.size = size

        assert isinstance(self.size, list)
        assert len(mean) == 3
        assert len(std) == 3
        assert len(size) == 2

    def __call__(self, filename):
        assert isinstance(filename, str)
        img = Image.open(filename).convert('RGB')
        img = transforms.Resize(self.size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.mean, self.std)(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        feature = self.model(img)
        return feature


class FeatureMatrix:
    """
    get feature matrix
    """

    def __init__(self, root, feature):
        assert isinstance(root, str)
        assert isinstance(feature, Feature)
        self.root = root
        self.feature = feature
        pass

    def __call__(self):
        path_list = os.listdir(self.root)
        result = []
        for i, image_path in tqdm(enumerate(path_list)):
            image_path_abs = os.path.join(self.root, image_path)
            feature = self.feature(image_path_abs)
            feature = feature.data.cpu().numpy()
            result.append([feature, image_path])
        return result


def main():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # configs
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # new add by gu
    cudnn.benchmark = True

    # models
    model = build_model(cfg, 5906)
    model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
    model = model.eval()
    model = model.cuda()

    feature = Feature(model=model, mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225], size=[384, 128])

    #
    TRAIN = "/home/xiangan/code_and_data/train_split/all/bounding_box_train"
    TEST_QUERY = "/home/xiangan/data_reid/testA/query_a"
    TEST_GALLERY = "/home/xiangan/data_reid/testA/gallery_a"
    OUTPUT = "/home/xiangan/reid_features"

    def feature_save(path, name):
        f = open(os.path.join(OUTPUT, name), 'wb')
        pickle.dump(FeatureMatrix(path, feature)(), f)

    feature_save(TRAIN, 'train.feat')
    feature_save(TEST_QUERY, 'query.feat')
    feature_save(TEST_GALLERY, 'gallery.feat')


if __name__ == '__main__':
    main()
