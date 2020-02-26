import glob
import os
from urllib.request import urlopen, Request

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_class_dirs(path):
    class_dirs = glob.glob(os.path.join(path, "*/"), recursive=True)
    return class_dirs


def url_to_img_bytes(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"}
    req = Request(url, headers=headers)

    img_http = urlopen(req)
    return img_http.read()


def url_to_img(url):
    img_bytes = url_to_img_bytes(url)
    img_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread(path):
    img = cv2.imread(path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def imshow(img, axis=True):
    plt.imshow(img)
    if not axis:
        plt.axis("off")
    plt.show()
