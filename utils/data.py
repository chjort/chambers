import glob
import os
from urllib.request import urlopen, Request

import tensorflow as tf


def list_subdirectories(path):
    return glob.glob(os.path.join(path, "*/"), recursive=True)


def read_lines(filepath):
    with open(filepath, "r") as f:
        data = f.read()
        lines = data.split("\n")
        lines = list(filter(None, lines))

    return lines


def read_classfile(classfile):
    class_dirs = read_lines(classfile)
    path_prefix = os.path.split(classfile)[0]
    class_dirs = [os.path.join(path_prefix, class_dir) + os.sep for class_dir in class_dirs]
    return class_dirs


def get_class_dirs(paths):
    if type(paths) is str:
        paths = [paths]

    all_class_dirs = []
    for p in paths:
        if os.path.isdir(p):
            class_dirs = list_subdirectories(p)
        elif os.path.isfile(p):
            class_dirs = read_classfile(p)
        else:
            raise ValueError("Invalid path. Must be a directory or a file.")

        all_class_dirs.extend(class_dirs)

    return all_class_dirs


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
