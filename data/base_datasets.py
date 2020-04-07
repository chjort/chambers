import tensorflow as tf

N_PARALLEL = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, input_dataset):
        self.dataset = input_dataset

    def apply(self, transformation_func):
        self.dataset = self.dataset.apply(transformation_func)

    def as_numpy_iterator(self):
        self.dataset = self.dataset.as_numpy_iterator()

    def batch(self, batch_size, drop_remainder=False):
        self.dataset = self.dataset.batch(batch_size, drop_remainder)

    def cache(self, filename=""):
        self.dataset = self.dataset.cache(filename)

    def concatenate(self, dataset):
        self.dataset = self.dataset.concatenate(dataset)

    def map(self, func):
        self.dataset = self.dataset.map(func, num_parallel_calls=N_PARALLEL)

    def flat_map(self, func):
        self.dataset = self.dataset.flat_map(func)

    def filter(self, predicate):
        self.dataset = self.dataset.filter(predicate)

    def interleave(self, map_func, cycle_length=-1, block_length=1):
        self.dataset = self.dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=N_PARALLEL)

    def padded_batch(self, batch_size, padded_shapes, padding_values=None, drop_remainder=False):
        self.dataset = self.dataset.padded_batch(batch_size, padded_shapes, padding_values, drop_remainder)

    def reduce(self, initial_state, reduce_func):
        self.dataset = self.dataset.reduce(initial_state, reduce_func)

    def unbatch(self):
        self.dataset = self.dataset.unbatch()

    def repeat(self, count=None):
        self.dataset = self.dataset.repeat(count)

    def prefetch(self, buffer_size):
        self.dataset = self.dataset.prefetch(buffer_size)

    def shard(self, num_shards, index):
        self.dataset = self.dataset.shard(num_shards, index)

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        self.dataset = self.dataset.shuffle(buffer_size, seed, reshuffle_each_iteration)

    def window(self, size, shift=None, stride=1, drop_remainder=False):
        self.dataset = self.dataset.window(size, shift, stride, drop_remainder)


class TensorDataset(Dataset):
    def __init__(self, element):
        input_dataset = tf.data.Dataset.from_tensors(element)
        super().__init__(input_dataset)


class TensorSliceDataset(Dataset):
    def __init__(self, element):
        input_dataset = tf.data.Dataset.from_tensor_slices(element)
        super().__init__(input_dataset)


class GeneratorDataset(Dataset):
    def __init__(self, element, output_types, output_shapes=None, args=None):
        input_dataset = tf.data.Dataset.from_generator(element, output_types, output_shapes, args)
        super().__init__(input_dataset)
