import random

if __name__ == '__main__':

    path = "c:\\Users\\Administrator\\Desktop\\labels_10w"

    count = 0
    origin_list = []
    for line in open(path, 'r'):
        count += 1
        print(count)
        modify_path = line.split('./')[1]
        path, p_id, probe_label = modify_path.split()
        origin_list.append((path, p_id, probe_label))

    p_id = None
    final = []
    for i in range(190000):
        if origin_list[i][1] == origin_list[i + 1][1]:
            final.append("%s,%s" % (origin_list[i][0], origin_list[i + 1][0]))
            print(origin_list[i], origin_list[i + 1])
        else:
            continue

    index_list = range(33, 19900)
    path_list = [x[0] for x in origin_list]
    pair_1 = path_list * 11
    pair_2 = path_list * 11
    random.shuffle(pair_1)
    random.shuffle(pair_2)
    for i in range(len(pair_1)):
        if pair_1[i] == pair_2[i]:
            pair_1[i] = pair_2[9]

    for i in range(len(pair_1)):
        if pair_1[i] == pair_2[i]:
            print('haha')
        final.append("%s,%s" % (pair_1[i], pair_2[i]))

    with open("pairs", 'w') as w:
        w.write("\n".join(final))