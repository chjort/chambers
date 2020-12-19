import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def rmac_regions(W, H, L):
    """
    Compute the RMAC regions for an input with width W and height W for L different scales.
    :param W: Width of input
    :param H: Height of input
    :param L: Number of scales for regions
    :return: List of tuples with format (x, y, w, h). Each tuple corresponds to a region.
    """
    ovr = 0.4  # desired overlap of neighboring regions
    steps = np.array(
        [2, 3, 4, 5, 6, 7], dtype=np.float
    )  # possible regions for the long dimension

    w = min(W, H)

    b = (max(H, W) - w) / (steps - 1)
    idx = np.argmin(
        abs(((w ** 2 - w * b) / w ** 2) - ovr)
    )  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd, Hd = 0, 0
    if H < W:
        Wd = idx + 1
    elif H > W:
        Hd = idx + 1

    regions = []
    for l in range(1, L + 1):
        wl = np.floor(2 * w / (l + 1))
        wl2 = np.floor(wl / 2 - 1)

        if (l + Wd - 1) == 0:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = np.floor(wl2 + np.arange(0, l + Wd) * b) - wl2  # center coordinates

        if (l + Hd - 1) == 0:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = np.floor(wl2 + np.arange(0, l + Hd) * b) - wl2  # center coordinates

        for i_ in cenH:
            for j_ in cenW:
                R = np.array(
                    [j_, i_, wl, wl], dtype=np.int
                )  # (W_offset, H_offset, W, H)
                if not min(R[2:]):
                    continue

                regions.append(R)

    regions = np.asarray(regions)
    return regions


class RMAC(tf.keras.layers.Layer):
    def __init__(self, scales=3, **kwargs):
        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {
            "channels_last",
            "channels_first",
        }, "dim_ordering must be in {channels_last, channels_first}"
        self.scales = scales
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == "channels_first":
            c = input_shape[1]
            w = input_shape[2]
            h = input_shape[3]
        elif self.dim_ordering == "channels_last":
            w = input_shape[1]
            h = input_shape[2]
            c = input_shape[3]
        else:
            raise ValueError("Invalid image format.")

        self.roi_boxes = tf.cast(rmac_regions(w, h, self.scales), tf.int32)
        self.n_rois = K.shape(self.roi_boxes)[0]
        self.n_channels = c

    def get_config(self):
        config = {"scales": self.scales}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        def batch_pool_rois(roi_box):
            ow = roi_box[0]
            oh = roi_box[1]
            tw = roi_box[2]
            th = roi_box[3]
            rois = tf.image.crop_to_bounding_box(
                x, offset_height=oh, offset_width=ow, target_height=th, target_width=tw
            )
            pooled_roi = self.pool_roi(rois)  # [batch_size, n_channels]
            return pooled_roi

        pooled_roi_batch = K.map_fn(
            batch_pool_rois, self.roi_boxes, dtype=tf.float32
        )  # [n_rois, batch_size, n_channels]
        pooled_roi_batch = tf.transpose(
            pooled_roi_batch, [1, 0, 2]
        )  # [batch_size, n_rois, n_channels]
        return pooled_roi_batch

    def pool_roi(self, roi):
        if self.dim_ordering == "channels_first":
            return K.max(roi, axis=(2, 3))
        elif self.dim_ordering == "channels_last":
            return K.max(roi, axis=(1, 2))
        else:
            raise ValueError("Invalid image format.")
