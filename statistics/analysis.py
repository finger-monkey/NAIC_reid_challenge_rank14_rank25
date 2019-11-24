import os

if __name__ == '__main__':
    #
    ROOT = "/home/xiangan/code_and_data/train_split/all_extra/bounding_box_train"
    image_path_list = os.listdir(ROOT)

    # persion id dict
    pid_dict = {}
    #
    for path in image_path_list:
        pid = path.split('_')[0]
        if pid not in pid_dict.keys():
            pid_dict[pid] = 1
        else:
            pid_dict[pid] += 1

    rich_sum = 0
    for k, v in pid_dict.items():
        if v > 100:
            print("pid:%s   image nums: %d" % (k, v))
            rich_sum += v
    print(rich_sum)
