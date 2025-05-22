import tensorflow as tf


def procrustes_tf(X, Y, validity_mask, allow_scaling=False, allow_reflection=False):
    """Register the points in Y by rotation, translation, uniform scaling (optional) and
    reflection (optional)
    to be closest to the corresponding points in X, in a least-squares sense.

    This function operates on batches. For each item in the batch a separate
    transform is computed independently of the others.

    Arguments:
       X: Tensor with shape [batch_size, n_points, point_dimensionality]
       Y: Tensor with shape [batch_size, n_points, point_dimensionality]
       validity_mask: Boolean Tensor with shape [batch_size, n_points] indicating
         whether a point is valid in Y
       allow_scaling: boolean, specifying whether uniform scaling is allowed
       allow_reflection: boolean, specifying whether reflections are allowed

    Returns the transformed version of Y.
    """
    meanY, T, output_scale, meanX = procrustes_tf_transf(
        X, Y, validity_mask, allow_scaling, allow_reflection)
    return ((Y - meanY) @ T) * output_scale + meanX


def procrustes_tf_transf(X, Y, validity_mask, allow_scaling=False, allow_reflection=False):
    validity_mask = validity_mask[..., tf.newaxis]
    _0 = tf.constant(0, X.dtype)
    n_points_per_example = tf.math.count_nonzero(
        validity_mask, axis=1, dtype=tf.float32, keepdims=True)
    denominator_correction_factor = tf.cast(
        tf.shape(validity_mask)[1], tf.float32) / n_points_per_example

    def normalize(Z):
        Z = tf.where(validity_mask, Z, _0)
        mean = tf.reduce_mean(Z, axis=1, keepdims=True) * denominator_correction_factor
        centered = tf.where(validity_mask, Z - mean, _0)
        norm = tf.norm(centered, axis=(1, 2), ord='fro', keepdims=True)
        normalized = centered / norm
        return mean, norm, normalized

    meanX, normX, normalizedX = normalize(X)
    meanY, normY, normalizedY = normalize(Y)
    A = tf.matmul(normalizedY, normalizedX, transpose_a=True)
    s, U, V = tf.linalg.svd(A)
    T = tf.matmul(U, V, transpose_b=True)
    s = tf.expand_dims(s, axis=-1)

    if allow_scaling:
        relative_scale = normX / normY
        output_scale = relative_scale * tf.reduce_sum(s, axis=1, keepdims=True)
    else:
        relative_scale = None
        output_scale = 1

    if not allow_reflection:
        # Check if T has a reflection component. If so, then remove it by flipping
        # across the direction of least variance, i.e. the last singular value/vector.
        has_reflection = (tf.linalg.det(T) < 0)[..., tf.newaxis, tf.newaxis]
        T_mirror = T - 2 * tf.matmul(U[..., -1:], V[..., -1:], transpose_b=True)
        T = tf.where(has_reflection, T_mirror, T)

        if allow_scaling:
            output_scale_mirror = output_scale - 2 * relative_scale * s[:, -1:]
            output_scale = tf.where(has_reflection, output_scale_mirror, output_scale)

    return meanY, T, output_scale, meanX
