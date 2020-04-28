import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.pooling import GlobalPooling2D


class GlobalGeneralizedMean(GlobalPooling2D):
    """
    Global Generalized Mean layer for spatial inputs

    This layer generalizes between max-pooling and average-pooling determined by the parameter p.
    The parameter p is trainable and can be learned from backpropagation.

    # References
        - [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)

    """

    def __init__(self, p=3, shared=True, trainable=True, data_format=None, **kwargs):
        super(GlobalGeneralizedMean, self).__init__(data_format=data_format, **kwargs)
        self._p_init = p
        self.shared = shared
        self.trainable = trainable

    def build(self, input_shape):
        if self.shared:
            p_shape = 1
        else:
            if self.data_format == 'channels_last':
                p_shape = input_shape[-1]
            else:
                p_shape = input_shape[1]

        self.p = self.add_weight(shape=[p_shape],
                                 initializer=initializers.constant(self._p_init),
                                 trainable=self.trainable
                                 )

    def call(self, inputs, **kwargs):
        x = tf.pow(inputs, self.p)
        if self.data_format == 'channels_last':
            x = tf.reduce_mean(x, axis=[1, 2])
        else:
            x = tf.reduce_mean(x, axis=[2, 3])
        x = tf.pow(x, 1 / self.p)

        return x

    def get_config(self):
        config = {'p': self._p_init, "shared": self.shared, 'trainable': self.trainable}
        base_config = super(GlobalGeneralizedMean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RoiPooling(Layer):
    """ROI pooling layer for 2D inputs.
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(batch_size, channels, rows, cols)` if dim_ordering='channels_first'
        or 4D tensor with shape:
        `(batch_size, rows, cols, channels)` if dim_ordering='channels_last'.
        X_roi:
        `(batch_size,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(batch_size, num_rois, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last',
                                     'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'

        super(RoiPooling, self).__init__(**kwargs)

    def call(self, x, mask=None):
        def batch_pool_rois(x):
            imgs = x[0]
            roi_boxes = x[1]
            return self.pool_image_rois(imgs, roi_boxes)

        return K.map_fn(batch_pool_rois, x, dtype=tf.float32)

    def pool_image_rois(self, x, roi_boxes):
        def crop_pool_roi(roi_box):
            roi_box = K.cast(roi_box, "int32")
            ow = roi_box[0]
            oh = roi_box[1]
            tw = roi_box[2]
            th = roi_box[3]
            roi = tf.image.crop_to_bounding_box(x, offset_height=oh, offset_width=ow,
                                                target_height=th, target_width=tw)
            pooled_roi = self.pool_roi(roi)
            return pooled_roi

        rois = K.map_fn(crop_pool_roi, roi_boxes, dtype=tf.float32)
        return rois

    def pool_roi(self, roi):
        if self.dim_ordering == "channels_first":
            return K.max(roi, axis=(1, 2))
        elif self.dim_ordering == "channels_last":
            return K.max(roi, axis=(0, 1))
        else:
            raise ValueError("Invalid image format.")


class RoiPooling_OG(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='channels_first'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='channels_last'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, num_rois, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last',
                                     'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'

        self.pool_list = pool_list
        self.num_rois = num_rois

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(RoiPooling_OG, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.nb_channels * self.num_outputs_per_channel

    def get_config(self):
        config = {'pool_list': self.pool_list, 'num_rois': self.num_rois}
        base_config = super(RoiPooling_OG, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        return self.pool_image_rois(x[0], x[1])

    def pool_image_rois(self, img, rois):
        outputs = []
        for roi_idx in range(self.num_rois):

            x = K.cast(rois[0, roi_idx, 0], "float32")
            y = K.cast(rois[0, roi_idx, 1], "float32")
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = K.cast([w / i for i in self.pool_list], "float32")
            col_length = K.cast([h / i for i in self.pool_list], "float32")

            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for ix in range(num_pool_regions):
                    for jy in range(num_pool_regions):
                        x1 = x + ix * col_length[pool_num]
                        x2 = x1 + col_length[pool_num]
                        y1 = y + jy * row_length[pool_num]
                        y2 = y1 + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        xm = self.crop_to_roi(img, x1, x2, y1, y2)
                        pooled_val = self.pool_roi(xm)
                        outputs.append(pooled_val)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.nb_channels * self.num_outputs_per_channel))

        return final_output

    def crop_to_roi(self, x, x1, x2, y1, y2):
        input_shape = K.shape(x)

        if self.dim_ordering == "channels_first":
            new_shape = [input_shape[0], input_shape[1], y2 - y1, x2 - x1]
            x_crop = x[:, :, y1:y2, x1:x2]
            xm = K.reshape(x_crop, new_shape)
        elif self.dim_ordering == "channels_last":
            new_shape = [input_shape[0], y2 - y1, x2 - x1, input_shape[3]]
            x_crop = x[:, y1:y2, x1:x2, :]
            xm = K.reshape(x_crop, new_shape)
        else:
            raise ValueError("Invalid image format.")
        return xm

    def pool_roi(self, roi):
        if self.dim_ordering == "channels_first":
            return K.max(roi, axis=(2, 3))
        elif self.dim_ordering == "channels_last":
            return K.max(roi, axis=(1, 2))
        else:
            raise ValueError("Invalid image format.")
