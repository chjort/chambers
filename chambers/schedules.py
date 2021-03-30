import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class LinearWarmup(LearningRateSchedule):
    def __init__(self, learning_rate, warmup_steps, ramp=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.ramp = ramp

        if ramp:
            learning_rate = self._get_learning_rate(0)
            self.step_size = learning_rate / warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        if self.ramp:
            learning_rate = tf.cond(
                pred=(step < self.warmup_steps),
                true_fn=lambda: step * self.step_size,
                false_fn=lambda: self._get_learning_rate(step - self.warmup_steps),
            )
        else:
            warmup_percent = step / self.warmup_steps
            lr_mult = tf.minimum(1.0, warmup_percent)
            learning_rate = self._get_learning_rate(step) * lr_mult

        return learning_rate

    def _get_learning_rate(self, step):
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate(step)
        elif callable(self.learning_rate):
            learning_rate = self.learning_rate()
        else:
            learning_rate = self.learning_rate

        return learning_rate

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "ramp": self.ramp,
        }
        return config
