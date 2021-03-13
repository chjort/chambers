import tensorflow as tf

from chambers import augmentations

IMG = tf.cast(
    [
        [139, 186, 208, 200],
        [175, 201, 198, 200],
        [166, 191, 193, 195],
        [124, 155, 172, 151],
    ],
    tf.uint8,
)
IMG = tf.stack([IMG, IMG, IMG], axis=-1)
IMG = tf.expand_dims(IMG, 0)

IMG_not_square = IMG[:, :, :3, :]


class TestAugmentations(tf.test.TestCase):
    def test_imagenet_normalization_caffe(self):
        target = tf.cast(
            [
                [35.060997, 82.061, 104.061, 96.061],
                [71.061, 97.061, 94.061, 96.061],
                [62.060997, 87.061, 89.061, 91.061],
                [20.060997, 51.060997, 68.061, 47.060997],
            ],
            tf.float32,
        )
        x = augmentations.ImageNetNormalization(mode="caffe")(IMG)
        x = x[0, ..., 0]

        self.assertAllEqual(x, target)

    def test_imagenet_normalization_tf(self):
        target = tf.cast(
            [
                [0.0901961327, 0.458823562, 0.631372571, 0.568627477],
                [0.372549057, 0.576470613, 0.552941203, 0.568627477],
                [0.301960826, 0.498039246, 0.513725519, 0.529411793],
                [-0.0274509788, 0.215686321, 0.349019647, 0.184313774],
            ],
            tf.float32,
        )
        x = augmentations.ImageNetNormalization(mode="tf")(IMG)
        x = x[0, ..., 0]

        self.assertAllEqual(x, target)

    def test_imagenet_normalization_torch(self):
        target = tf.cast(
            [
                [0.262436897, 1.06730032, 1.44404483, 1.30704677],
                [0.878928, 1.32417154, 1.27279735, 1.30704677],
                [0.724805236, 1.15292406, 1.1871736, 1.22142303],
                [0.00556548592, 0.536432922, 0.827553749, 0.467933923],
            ],
            tf.float32,
        )
        x = augmentations.ImageNetNormalization(mode="torch")(IMG)
        x = x[0, ..., 0]

        self.assertAllEqual(x, target)

    def test_resize_min(self):
        x = augmentations.ResizingMinMax(min_side=100)(IMG_not_square)
        self.assertAllEqual(x.shape, [1, 133, 100, 3])

    def test_resize_max(self):
        x = augmentations.ResizingMinMax(max_side=100)(IMG_not_square)
        self.assertAllEqual(x.shape, [1, 100, 75, 3])

    def test_resize_min_max0(self):
        x = augmentations.ResizingMinMax(min_side=100, max_side=100)(IMG_not_square)
        self.assertAllEqual(x.shape, [1, 100, 75, 3])

    def test_resize_min_max1(self):
        x = augmentations.ResizingMinMax(min_side=100, max_side=50)(IMG_not_square)
        self.assertAllEqual(x.shape, [1, 50, 37, 3])
