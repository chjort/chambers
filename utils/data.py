import glob
import os
from urllib.request import urlopen, Request

import tensorflow as tf


def get_class_dirs(path):
    class_dirs = glob.glob(os.path.join(path, "*/"), recursive=True)
    return class_dirs


def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"}
    req = Request(url, headers=headers)

    img_file = urlopen(req)
    return img_file


def url_to_img_bytes(url):
    img_file = download_image(url)
    return img_file.read()


def url_to_img(url):
    img_bytes = url_to_img_bytes(url)
    return tf.io.decode_image(img_bytes)
