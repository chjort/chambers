from .base_datasets import N_PARALLEL

class ImageLabelMixin:
    def map_image(self, func, *args, **kwargs):
        def fn(images, labels):
            return func(images, *args, **kwargs), labels

        self.dataset = self.dataset.map(fn, num_parallel_calls=N_PARALLEL)