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


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # new add by gu
    cudnn.benchmark = True
    # model = train(cfg)
    model = build_model(cfg, 5906)
    model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))

    test_root = cfg.DATASETS.ROOT_DIR
    print(test_root)

    model = model.eval()
    model = model.cuda()

    result = []

    # query_root = "/home/xiangan/data_reid/testA/query_a"
    query_root = os.path.join(test_root, 'query_a')
    query_path = os.listdir(query_root)
    for i in range(len(query_path)):
        print(i)
        name = os.path.join(query_root, query_path[i])
        feature = get_image(name, model)
        feature = feature.data.cpu().numpy()
        result.append([feature, query_path[i]])
    pickle.dump(result, open(cfg.OUTPUT_DIR + '/query_a_feature.feat', 'wb'))

    result = []
    # gallery_root = "/home/xiangan/data_reid/testA/gallery_a"
    gallery_root = os.path.join(test_root, 'gallery_a')
    gallery_path = os.listdir(gallery_root)
    for i in range(len(gallery_path)):
        print(i)
        name = os.path.join(gallery_root, gallery_path[i])
        feature = get_image(name, model)
        feature = feature.data.cpu().numpy()
        result.append([feature, gallery_path[i]])
    pickle.dump(result, open(cfg.OUTPUT_DIR + '/gallery_a_feature.feat', 'wb'))

    return model


def get_image(filename, model):
    img = Image.open(filename).convert('RGB')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = transforms.Resize([384, 128])(img)
    # img2 = transforms.RandomHorizontalFlip(p=1.0)(img)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean, std)(img)
    img = img.unsqueeze(0)
    img = img.cuda()

    # img2 = transforms.ToTensor()(img2)
    # img2 = transforms.Normalize(mean, std)(img2)
    # img2 = img2.unsqueeze(0)
    # img2 = img2.cuda()

    feature = model(img)
    # feature2 = model(img2)

    # feature = torch.cat((feature, feature2), 1)

    return feature


if __name__ == '__main__':
    model = main()
