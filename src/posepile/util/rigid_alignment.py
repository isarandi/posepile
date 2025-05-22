import numpy as np

from simplepyutils.argparse import logger


def rigid_align(coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
                reflection_align=False):
    """Returns the predicted coordinates after rigid alignment to the ground truth."""

    if joint_validity_mask is None:
        joint_validity_mask = np.ones_like(coords_pred[..., 0], dtype=bool)

    valid_coords_pred = coords_pred[joint_validity_mask]
    valid_coords_true = coords_true[joint_validity_mask]
    try:
        d, Z, tform = procrustes(
            valid_coords_true, valid_coords_pred, scaling=scale_align,
            reflection='best' if reflection_align else False)
    except np.linalg.LinAlgError:
        logger.error('Cannot do Procrustes alignment, returning original prediction.')
        return coords_pred

    T = tform['rotation']
    b = tform['scale']
    c = tform['translation']
    return b * coords_pred @ T + c


def rigid_align_many(
        coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
        reflection_align=False):
    if joint_validity_mask is None:
        joint_validity_mask = np.ones_like(coords_pred[..., 0], dtype=bool)

    return np.stack([
        rigid_align(p, t, joint_validity_mask=jv, scale_align=scale_align,
                    reflection_align=reflection_align)
        for p, t, jv in zip(coords_pred, coords_true, joint_validity_mask)])


def procrustes(X, Y, scaling=True, reflection='best'):
    # From https://github.com/prlz77/prlz77.cvtools
    """A port of MATLAB's `procrustes` function to Numpy.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros((n, m - my))), axis=0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform

