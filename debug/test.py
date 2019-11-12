import cv2
import os

import datetime
starttime = datetime.datetime.now()

path = "/home/xiangan/test/gallery"
gallery_list = os.listdir(path)
for g in gallery_list:
    _ = cv2.imread(g)
endtime = datetime.datetime.now()

print((endtime - starttime).seconds)
