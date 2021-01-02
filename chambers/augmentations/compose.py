class Compose:
    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(self, *args):
        x = self.augmentations[0](*args)

        for fn in self.augmentations[1:]:
            x = fn(*x)

        return x
