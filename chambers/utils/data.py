import tensorflow as tf

from chambers.utils.generic import ProgressBar


def valid_cardinality(dataset):
    if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
        return False
    if dataset.cardinality() == tf.data.UNKNOWN_CARDINALITY:
        return False
    return True


def _to_dataset(x, y=None, n=None):
    if not isinstance(x, tf.data.Dataset):
        n = tf.shape(x)[0]
        if y is not None:
            x = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            x = tf.data.Dataset.from_tensor_slices(x)
    else:
        if valid_cardinality(x):
            n = x.cardinality()
        elif not valid_cardinality(x) and n is None:
            raise ValueError("Unable to infer length of dataset {}.".format(x))

    return x, n


def pair_iteration_dataset(q, c, bq, bc, yq=None, yc=None, nq=None, nc=None):
    qd, nq = _to_dataset(q, yq, nq)
    cd, nc = _to_dataset(c, yc, nc)
    with_labels = not isinstance(qd.element_spec, tf.TensorSpec)

    bq = tf.cast(bq, tf.int64)
    bc = tf.cast(bc, tf.int64)
    nq = tf.cast(nq, tf.int64) if nq is not None else nq
    nc = tf.cast(nc, tf.int64) if nc is not None else nc

    qd = qd.batch(bq)
    cd = cd.batch(bc)

    nqb = tf.cast(tf.math.ceil(nq / bq), tf.int64)
    ncb = tf.cast(tf.math.ceil(nc / bc), tf.int64)

    if with_labels:
        repeat_batch = lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(ncb)
    else:
        repeat_batch = lambda x: tf.data.Dataset.from_tensors(x).repeat(ncb)

    qd = qd.flat_map(repeat_batch)
    cd = cd.repeat(nqb)

    if with_labels:
        td = tf.data.Dataset.zip((qd, cd))
        # ((x_q, x_c), (y_q, y_c))
        td = td.map(lambda q, c: ((q[0], c[0]), (q[1], c[1])))
    else:
        td = tf.data.Dataset.zip(((qd, cd),))

    return td


def reshape_pair_predictions(x, bq, bc, nq, nc, y=None):
    nqb = tf.cast(tf.math.ceil(nq / bq), tf.int64)
    ncb = tf.cast(tf.math.ceil(nc / bc), tf.int64)

    x = tf.reshape(x, [nqb, ncb, bq, bc])
    x = tf.transpose(x, [0, 2, 1, 3])  # [nqb, bq, ncb, bc]
    x = tf.reshape(x, [nq, nc])

    if y is not None:
        yq, yc = y
        yq = tf.reshape(yq, [nqb, ncb, bq])[:, 0]
        yq = tf.reshape(yq, [-1, 1])
        yc = yc[:nc]
        return x, (yq, yc)

    return x


def batch_predict_pairs(
    model, q, bq, c=None, bc=None, yq=None, yc=None, nq=None, nc=None, verbose=True
):
    if c is None:
        c = q
        bc = bq
        yc = yq
        nc = nq
    elif bc is None:
        bc = bq

    q, nq = _to_dataset(q, yq, nq)
    c, nc = _to_dataset(c, yc, nc)

    bq = tf.cast(bq, tf.int64)
    bc = tf.cast(bc, tf.int64)
    nq = tf.cast(nq, tf.int64) if nq is not None else nq
    nc = tf.cast(nc, tf.int64) if nc is not None else nc

    bq = tf.minimum(bq, nq)
    bc = tf.minimum(bc, nc)

    td = pair_iteration_dataset(q, c, bq, bc, yq, yc, nq, nc)

    if verbose:
        nqb = tf.cast(tf.math.ceil(nq / bq), tf.int32)
        ncb = tf.cast(tf.math.ceil(nc / bc), tf.int32)

        prog = ProgressBar(total=nqb * ncb)
        td = td.apply(prog.dataset_apply_fn)

    z = model.predict(td)

    if isinstance(z, tuple):
        z, y = z
        z, y = reshape_pair_predictions(z, bq, bc, nq, nc, y=y)
        return z, y

    z = reshape_pair_predictions(z, bq, bc, nq, nc)
    return z
