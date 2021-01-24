import re
from typing import Union, Callable, Type

import tensorflow as tf
from tensorflow_addons.optimizers import DecoupledWeightDecayExtension
from tensorflow_addons.utils.types import FloatTensorLike
from typeguard import typechecked


class WeightDecayExtension(DecoupledWeightDecayExtension):
    """This class allows to extend optimizers with decoupled weight decay.

    It implements the decoupled weight decay described by Loshchilov & Hutter
    (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
    decoupled from the optimization steps w.r.t. to the loss function.
    For SGD variants, this simplifies hyperparameter search since it decouples
    the settings of weight decay and learning rate.
    For adaptive gradient algorithms, it regularizes variables with large
    gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.

    This class alone is not an optimizer but rather extends existing
    optimizers with decoupled weight decay. We explicitly define the two
    examples used in the above paper (SGDW and AdamW), but in general this can
    extend any OptimizerX class by using
        `ExtendedCls = extend_with_decoupled_weight_decay(OptimizerX)`.
    Weight decay can then be set when instantiating the optimizer:
        `optimizerX = ExtendedCls(weight_decay=0.001, learning_rate=0.001)`.
    In order for it to work, it must be the first class the Optimizer with
    weight decay inherits from, e.g.

    ```python
    class AdamW(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
      def __init__(self, weight_decay, *args, **kwargs):
        super(AdamW, self).__init__(weight_decay, *args, **kwargs).
    ```

    Note: this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of'var' in the update step!

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:

    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    # ...

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```
    """

    @typechecked
    def __init__(
        self,
        weight_decay: Union[FloatTensorLike, Callable],
        decay_include=None,
        decay_exclude=None,
        **kwargs,
    ):
        """Extension class that adds weight decay to an optimizer.

        Args:
            weight_decay: A `Tensor`, a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                to decay the variable by, in the update step.
            **kwargs: Optional list or tuple or set of `Variable` objects to
                decay.
        """
        if decay_include is not None and decay_exclude is not None:
            raise ValueError(
                "Got both `decay_include` and `decay_exclude` arguments. Use only `decay_include` or `decay_exclude`."
            )

        wd = kwargs.pop("weight_decay", weight_decay)
        super().__init__(weight_decay=weight_decay, **kwargs)
        self.decay_include = list(decay_include) if decay_include is not None else None
        self.decay_exclude = list(decay_exclude) if decay_exclude is not None else None
        self._set_hyper("weight_decay", wd)

    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        """Minimize `loss` by updating `var_list`.

        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before
        applying then call `tf.GradientTape` and `apply_gradients()` explicitly
        instead of using this function.

        Args:
            loss: `Tensor` or callable. If a callable, `loss` should take no
                arguments and return the value to minimize. If a `Tensor`, the
                `tape` argument must be passed.
            var_list: list or tuple of `Variable` objects to update to
                minimize `loss`, or a callable returning the list or tuple of
                `Variable` objects. Use callable when the variable list would
                otherwise be incomplete before `minimize` since the variables
                are created at the first time `loss` is called.
            grad_loss: Optional. A `Tensor` holding the gradient computed for
                `loss`.
            name: Optional name for the returned operation.
            tape: (Optional) `tf.GradientTape`. If `loss` is provided as a
                `Tensor`, the tape that computed the `loss` must be provided.
        Returns:
            An Operation that updates the variables in `var_list`.
        Raises:
            ValueError: If some of the variables are not `Variable` objects.
        """

        # NOTE: overwriting DecoupledWeightDecayExtension.minimize to not modify self._decay_var_list
        #  Now directly calling .minimize method of the optimizer
        return super(DecoupledWeightDecayExtension, self).minimize(
            loss, var_list=var_list, grad_loss=grad_loss, name=name, tape=tape
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.
            name: Optional name for the returned operation.  Default to the
                name passed to the `Optimizer` constructor.
            **kwargs: Additional arguments to pass to the base optimizer's
                apply_gradient method, e.g., TF2.2 added an argument
                `experimental_aggregate_gradients`.
        Returns:
            An `Operation` that applies the specified gradients.
        Raises:
            TypeError: If `grads_and_vars` is malformed.
            ValueError: If none of the variables have gradients.
        """

        # NOTE: overwriting DecoupledWeightDecayExtension.apply_gradients to not modify self._decay_var_list
        #  Now directly calling .apply_gradients method of the optimizer
        return super(DecoupledWeightDecayExtension, self).apply_gradients(
            grads_and_vars, name=name, **kwargs
        )

    def _decay_weights_op(self, var, apply_state=None):
        # if not self.decay_include or var.ref() in self.decay_include:
        if self._is_decay_allowed(var):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            return var.assign_sub(coefficients["wd_t"] * var, self._use_locking)
        return tf.no_op()

    def _decay_weights_sparse_op(self, var, indices, apply_state=None):
        # if not self.decay_include or var.ref() in self.decay_include:
        if self._is_decay_allowed(var):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)

            update = -coefficients["wd_t"] * tf.gather(var, indices)
            return self._resource_scatter_add(var, indices, update)
        return tf.no_op()

    def _is_decay_allowed(self, var):
        if self.decay_include is not None:
            for name in self.decay_include:
                if re.search(name, var.name) is not None:
                    return True
            return False
        else:
            for name in self.decay_exclude:
                if re.search(name, var.name) is not None:
                    return False
            return True

    def get_config(self):
        config = {
            "decay_include": self.decay_include,
            "decay_exclude": self.decay_exclude,
        }
        base_config = super(WeightDecayExtension, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@typechecked
def extend_with_weight_decay(
    base_optimizer: Type[tf.keras.optimizers.Optimizer],
) -> Type[tf.keras.optimizers.Optimizer]:
    """Factory function returning an optimizer class with decoupled weight
    decay.

    Returns an optimizer class. An instance of the returned class computes the
    update step of `base_optimizer` and additionally decays the weights.
    E.g., the class returned by
    `extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)` is
    equivalent to `tfa.optimizers.AdamW`.

    The API of the new optimizer class slightly differs from the API of the
    base optimizer:
    - The first argument to the constructor is the weight decay rate.
    - `minimize` and `apply_gradients` accept the optional keyword argument
      `decay_var_list`, which specifies the variables that should be decayed.
      If `None`, all variables that are optimized are decayed.

    Usage example:
    ```python
    # MyAdamW is a new class
    MyAdamW = extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
    # Create a MyAdamW object
    optimizer = MyAdamW(weight_decay=0.001, learning_rate=0.001)
    # update var1, var2 but only decay var1
    optimizer.minimize(loss, var_list=[var1, var2], decay_variables=[var1])

    Note: this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of 'var' in the update step!

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:

    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    # ...

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```

    Note: you might want to register your own custom optimizer using
    `tf.keras.utils.get_custom_objects()`.

    Args:
        base_optimizer: An optimizer class that inherits from
            tf.optimizers.Optimizer.

    Returns:
        A new optimizer class that inherits from DecoupledWeightDecayExtension
        and base_optimizer.
    """

    class OptimizerWithWeightDecay(WeightDecayExtension, base_optimizer):
        """Base_optimizer with decoupled weight decay.

        This class computes the update step of `base_optimizer` and
        additionally decays the variable with the weight decay being
        decoupled from the optimization steps w.r.t. to the loss
        function, as described by Loshchilov & Hutter
        (https://arxiv.org/pdf/1711.05101.pdf). For SGD variants, this
        simplifies hyperparameter search since it decouples the settings
        of weight decay and learning rate. For adaptive gradient
        algorithms, it regularizes variables with large gradients more
        than L2 regularization would, which was shown to yield better
        training loss and generalization error in the paper above.
        """

        @typechecked
        def __init__(
            self,
            weight_decay: Union[FloatTensorLike, Callable],
            decay_include=None,
            decay_exclude=None,
            *args,
            **kwargs,
        ):
            # super delegation is necessary here
            super().__init__(
                weight_decay,
                decay_include=decay_include,
                decay_exclude=decay_exclude,
                *args,
                **kwargs,
            )

    return OptimizerWithWeightDecay


@tf.keras.utils.register_keras_serializable(package="Chambers")
class SGDW(WeightDecayExtension, tf.keras.optimizers.SGD):
    """Optimizer that implements the Momentum algorithm with weight_decay.

    This is an implementation of the SGDW optimizer described in "Decoupled
    Weight Decay Regularization" by Loshchilov & Hutter
    (https://arxiv.org/abs/1711.05101)
    ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
    It computes the update step of `tf.keras.optimizers.SGD` and additionally
    decays the variable. Note that this is different from adding
    L2 regularization on the variables to the loss. Decoupling the weight decay
    from other hyperparameters (in particular the learning rate) simplifies
    hyperparameter search.

    For further information see the documentation of the SGD Optimizer.

    This optimizer can also be instantiated as
    ```python
    extend_with_decoupled_weight_decay(tf.keras.optimizers.SGD,
                                       weight_decay=weight_decay)
    ```

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:

    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    # ...

    optimizer = tfa.optimizers.SGDW(
        learning_rate=lr, weight_decay=wd, momentum=0.9)
    ```
    """

    @typechecked
    def __init__(
        self,
        weight_decay: Union[FloatTensorLike, Callable],
        decay_include=None,
        decay_exclude=None,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        momentum: Union[FloatTensorLike, Callable] = 0.0,
        nesterov: bool = False,
        name: str = "SGDW",
        **kwargs,
    ):
        """Construct a new SGDW optimizer.

        For further information see the documentation of the SGD Optimizer.

        Args:
            learning_rate: float hyperparameter >= 0. Learning rate.
            momentum: float hyperparameter >= 0 that accelerates SGD in the
                relevant direction and dampens oscillations.
            nesterov: boolean. Whether to apply Nesterov momentum.
            name: Optional name prefix for the operations created when applying
                gradients.  Defaults to 'SGD'.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse decay
                of learning rate. `lr` is included for backward compatibility,
                recommended to use `learning_rate` instead.
        """
        super().__init__(
            weight_decay=weight_decay,
            decay_include=decay_include,
            decay_exclude=decay_exclude,
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            name=name,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class AdamW(WeightDecayExtension, tf.keras.optimizers.Adam):
    """Optimizer that implements the Adam algorithm with weight decay.

    This is an implementation of the AdamW optimizer described in "Decoupled
    Weight Decay Regularization" by Loshch ilov & Hutter
    (https://arxiv.org/abs/1711.05101)
    ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).

    It computes the update step of `tf.keras.optimizers.Adam` and additionally
    decays the variable. Note that this is different from adding L2
    regularization on the variables to the loss: it regularizes variables with
    large gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.

    For further information see the documentation of the Adam Optimizer.

    This optimizer can also be instantiated as
    ```python
    extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam,
                                       weight_decay=weight_decay)
    ```

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:

    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    # ...

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```
    """

    @typechecked
    def __init__(
        self,
        weight_decay: Union[FloatTensorLike, Callable],
        decay_include=None,
        decay_exclude=None,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: Union[FloatTensorLike, Callable] = 0.9,
        beta_2: Union[FloatTensorLike, Callable] = 0.999,
        epsilon: FloatTensorLike = 1e-07,
        amsgrad: bool = False,
        name: str = "AdamW",
        **kwargs,
    ):
        """Construct a new AdamW optimizer.

        For further information see the documentation of the Adam Optimizer.

        Args:
            weight_decay: A Tensor or a floating point value. The weight decay.
            learning_rate: A Tensor or a floating point value. The learning
                rate.
            beta_1: A float value or a constant float tensor. The exponential
                decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor. The exponential
                decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just
                before Section 2.1), not the epsilon in Algorithm 1 of the
                paper.
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            name: Optional name for the operations created when applying
                gradients. Defaults to "AdamW".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse decay
                of learning rate. `lr` is included for backward compatibility,
                recommended to use `learning_rate` instead.
        """
        super().__init__(
            weight_decay=weight_decay,
            decay_include=decay_include,
            decay_exclude=decay_exclude,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )
