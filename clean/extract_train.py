import argparse
import os
import pickle
import sys

import torch
from PIL import Image
from torch.backends import cudnn
from torchvision import transforms

sys.path.append('../')
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

    dataset_root = "/home/xiangan/code_and_data/train_split/all_extra/bounding_box_train"
    image_path = os.listdir(dataset_root)

    for i, path in enumerate(image_path):
        if i % 1000 == 0:
            print(i)
        name = os.path.join(dataset_root, path)
        feature = get_image(name, model)
        feature = feature.data.cpu().numpy()
        result.append([feature, path])
    pickle.dump(result, open("/home/xiangan/dgreid/features/trainset_features_056/trainset.feat", 'wb'))


def get_image(filename, model):
    img = Image.open(filename).convert('RGB')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = transforms.Resize([384, 128])(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean, std)(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    feature = model(img)

    return feature


if __name__ == '__main__':
    model = main()
