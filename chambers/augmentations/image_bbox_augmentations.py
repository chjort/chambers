import tensorflow as tf

from chambers.augmentations.single_image_augmentations import _resize_min_max as _resize


def random_size_crop_WIP(img, boxes, labels, min_size, max_size):
    img_shape = tf.shape(img)
    img_h = img_shape[0]
    img_w = img_shape[0]
    img_area = img_h * img_w

    hw = tf.random.uniform([2], min_size, max_size + 1, dtype=tf.int32)
    h = hw[0]
    w = hw[1]

    min_dim = tf.minimum(h, w)
    max_dim = tf.maximum(h, w)
    aspect_ratio = [min_dim / max_dim, max_dim / min_dim]

    crop_area = h * w
    area_ratio = crop_area / img_area

    begin, size, bboxes = tf.image.sample_distorted_bounding_box(
        img_shape,
        [boxes],
        min_object_covered=0.1,
        aspect_ratio_range=aspect_ratio,
        area_range=[area_ratio, area_ratio + 0.005],
    )
    return begin, size, bboxes


def random_size_crop(img, boxes, labels, min_size, max_size):
    """

    :param img:
    :type img:
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y0, x0]
    :type boxes:
    :param labels: 2-D Tensor of shape (box_number, 1) containing class labels for each box.
    :type labels:
    :param min_size:
    :type min_size:
    :param max_size:
    :type max_size:
    :return:
    :rtype:
    """

    input_shape = tf.shape(img)
    img_h = input_shape[0]
    img_w = input_shape[1]

    # TODO: Crop such that there is always a bounding box in the crop
    #   or filter empty boxes in dataset iterator

    hw = tf.random.uniform([2], min_size, max_size + 1, dtype=tf.int32)
    h = hw[0]
    w = hw[1]

    if h >= img_h:
        h = img_h
        y0 = 0
    else:
        y0 = tf.random.uniform([], 0, img_h - h, dtype=tf.int32)

    if w >= img_w:
        w = img_w
        x0 = 0
    else:
        x0 = tf.random.uniform([], 0, img_w - w, dtype=tf.int32)

    y0_n = y0 / img_h
    x0_n = x0 / img_w
    h_n = h / img_h
    w_n = w / img_w

    boxes = box_normalize_yxyx(boxes, img)
    boxes_cropped = _clip_bboxes(boxes, y0_n, x0_n, h_n, w_n)
    zero_area_box = _zero_area_boxes_mask(boxes_cropped)

    boxes_cropped = tf.boolean_mask(boxes_cropped, zero_area_box)
    labels = tf.boolean_mask(labels, zero_area_box)

    img_cropped = tf.image.crop_to_bounding_box(img, y0, x0, h, w)
    boxes_cropped = box_denormalize_yxyx(boxes_cropped, img_cropped)

    return img_cropped, boxes_cropped, labels


@tf.function
def flip_up_down(img, boxes):
    """
    Flip an image and bounding boxes vertically (upside down).
    :param img: 3-D Tensor of shape [height, width, channels]
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y0, x0]
    :return: image, bounding boxes
    """
    h = tf.cast(tf.shape(img)[0], tf.float32)
    with tf.name_scope("flip_up_down"):
        boxes = boxes * tf.constant([-1, 1, -1, 1], dtype=tf.float32) + tf.stack(
            [h, 0.0, h, 0.0]
        )
        boxes = tf.stack([boxes[:, 2], boxes[:, 1], boxes[:, 0], boxes[:, 3]], axis=1)

        img = tf.image.flip_up_down(img)

    return img, boxes


@tf.function
def flip_left_right(img, boxes):
    """
    Flip an image and bounding boxes horizontally (left to right).
    :param img: 3-D Tensor of shape [height, width, channels]
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y0, x0]
    :return: image, bounding boxes
    """
    w = tf.cast(tf.shape(img)[1], tf.float32)
    with tf.name_scope("flip_left_right"):
        boxes = boxes * tf.constant([1, -1, 1, -1], dtype=tf.float32) + tf.stack(
            [0.0, w, 0.0, w]
        )
        boxes = tf.stack([boxes[:, 0], boxes[:, 3], boxes[:, 2], boxes[:, 1]], axis=1)

        img = tf.image.flip_left_right(img)

    return img, boxes


def random_flip_left_right(img, boxes):
    img, boxes = tf.cond(
        tf.random.uniform([1], 0, 1) > 0.5,
        true_fn=lambda: flip_left_right(img, boxes),
        false_fn=lambda: (img, boxes),
    )
    return img, boxes


def random_flip_up_down(img, boxes):
    img, boxes = tf.cond(
        tf.random.uniform([1], 0, 1) > 0.5,
        true_fn=lambda: flip_up_down(img, boxes),
        false_fn=lambda: (img, boxes),
    )
    return img, boxes


def resize(img, boxes, shape=None, min_side=800, max_side=1333):
    """


    :param img: image with shape [h, w, c]
    :type img:
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y1, x1]
    :type boxes:
    :param min_side:
    :type min_side:
    :param max_side:
    :type max_side:
    :return:
    :rtype:
    """
    if shape is None:
        imgr = _resize(img, min_side=min_side, max_side=max_side)
    else:
        imgr = tf.image.resize(img, shape)

    img_hw = tf.shape(img)[:2]
    imgr_hw = tf.shape(imgr)[:2]

    hw_ratios = tf.cast(imgr_hw / img_hw, tf.float32)
    h_ratio = hw_ratios[0]
    w_ratio = hw_ratios[1]

    boxr = boxes * tf.stack([h_ratio, w_ratio, h_ratio, w_ratio])  # [y0, x0, y1, x1]

    # [x0, y0, x1, y1], or [x0, y0, w, h] or [center_x, center_y, w, h]
    # boxr = boxes * tf.stack([w_ratio, h_ratio, w_ratio, h_ratio])

    return imgr, boxr


def random_resize_min(img, boxes, min_sides, max_side=1333):
    min_sides = tf.convert_to_tensor(min_sides)

    rand_idx = tf.random.uniform(
        [1], minval=0, maxval=tf.shape(min_sides)[0], dtype=tf.int32
    )[0]
    min_side = min_sides[rand_idx]
    return resize(img, boxes, min_side=min_side, max_side=max_side)


def box_normalize_xyxy(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_xyxy(boxes, h, w)


def box_normalize_yxyx(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_yxyx(boxes, h, w)


def box_normalize_xywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_xyxy(boxes, h, w)


def box_normalize_cxcywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_normalize_xyxy(boxes, h, w)


def box_denormalize_xyxy(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_xyxy(boxes, h, w)


def box_denormalize_yxyx(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_yxyx(boxes, h, w)


def box_denormalize_xywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_xyxy(boxes, h, w)


def box_denormalize_cxcywh(boxes, img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    return _box_denormalize_xyxy(boxes, h, w)


def _box_normalize_xyxy(boxes, img_h, img_w):
    """
    Normalizes bounding box to have coordinates between 0 and 1. Bounding boxes are expected to
    have format [x0, y0, x1, y1].

    :param boxes: List of bounding boxes each with format [x0, y0, x1, y1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes / tf.cast(
        tf.stack([img_w, img_h, img_w, img_h]), tf.float32
    )  # boxes / [w, h, w, h]
    return boxes


def _box_normalize_yxyx(boxes, img_h, img_w):
    """
    Normalizes bounding box to have coordinates between 0 and 1. Bounding boxes are expected to
    have format [y0, x0, y1, x1].

    :param boxes: List of bounding boxes each with format [y0, x0, y1, x1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes / tf.cast(
        tf.stack([img_h, img_w, img_h, img_w]), tf.float32
    )  # boxes / [h, w, h, w]
    return boxes


def _box_denormalize_xyxy(boxes, img_h, img_w):
    """
    Denormalizes bounding box with coordinates between 0 and 1 to have coordinates in original image height and width.
    Bounding boxes are expected to have format [x0, y0, x1, y1].

    :param boxes: List of bounding boxes each with format [x0, y0, x1, y1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes * tf.cast(
        tf.stack([img_w, img_h, img_w, img_h]), tf.float32
    )  # boxes * [w, h, w, h]
    return boxes


def _box_denormalize_yxyx(boxes, img_h, img_w):
    """
    Denormalizes bounding box with coordinates between 0 and 1 to have coordinates in original image height and width.
    Bounding boxes are expected to have format [y0, x0, y1, x1].

    :param boxes: List of bounding boxes each with format [y0, x0, y1, x1]
    :type boxes: list[list] or tensorflow.Tensor
    :param img_h: The height of the image the bounding box belongs to.
    :type img_h: int or tensorflow.Tensor
    :param img_w: The width of the image the bounding box belongs to.
    :type img_w: int or tensorflow.Tensor
    :return: Normalized bounding boxes.
    :rtype: tensorflow.Tensor
    """

    boxes = boxes * tf.cast(
        tf.stack([img_h, img_w, img_h, img_w]), tf.float32
    )  # boxes * [h, w, h, w]
    return boxes


def _clip_bboxes(boxes_relative, y0, x0, h, w):
    """
    Calculates new coordinates for given bounding boxes given the cut area of an image.
    :param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y0, x0] with values
        between 0 and 1.
    :param y0: Relative clipping coordinate.
    :param x0: Relative clipping coordinate.
    :param h: Relative clipping height.
    :param w: Relative clipping width.
    :return: clipped bounding boxes
    """
    # move the coordinates according to new min value
    bboxes_move_min = tf.stack([y0, x0, y0, x0])
    bboxes = boxes_relative - tf.cast(bboxes_move_min, boxes_relative.dtype)

    # if we use relative coordinates, we have to scale the coordinates to be between 0 and 1 again
    bboxes_scale = tf.stack([h, w, h, w])
    bboxes = bboxes / tf.cast(bboxes_scale, dtype=bboxes.dtype)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    return bboxes


def _zero_area_boxes_mask(boxes):
    boxes = tf.reshape(boxes, [-1, 2, 2])
    zero_area_boxes = tf.reduce_all(boxes[:, 1, :] > boxes[:, 0, :], axis=1)
    boxes = tf.reshape(boxes, [-1, 4])
    zero_area_boxes.set_shape([None])

    # boxes = tf.boolean_mask(boxes, zero_area_boxes)
    return zero_area_boxes


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
def _check_boxes_in_img_deprecated(boxes):
    """

    :param boxes:  2-D Tensor of shape (box_number, 4) containing bounding boxes in format [y0, x0, y0, x0] with values
        between 0 and 1.
    :return:
    :rtype:
    """
    y_in_img = tf.logical_and(
        tf.logical_or(boxes[:, 0] != 1, boxes[:, 0] != 0), boxes[:, 0] != boxes[:, 2]
    )

    x_in_img = tf.logical_and(
        tf.logical_or(boxes[:, 1] != 1, boxes[:, 1] != 0), boxes[:, 1] != boxes[:, 3]
    )

    box_in_img = tf.logical_and(y_in_img, x_in_img)
    return box_in_img
