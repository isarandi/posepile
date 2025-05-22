import numpy as np
import scipy.io
from addict import Addict


def load_mat(path):
    """This function is better than scipy.io.loadmat as it cures the problem of not properly
    recovering Python dictionaries from mat files. It transforms all entries which are still
    mat-objects.
    """
    dic = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    return Addict({k: _cure(v) for k, v in dic.items()})


def _to_addict(mat_struct):
    return Addict(
        {field_name: _cure(getattr(mat_struct, field_name))
         for field_name in mat_struct._fieldnames})


def _to_list(ndarray):
    return [_cure(elem) for elem in ndarray]


def _cure(elem):
    if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
        return _to_addict(elem)
    elif isinstance(elem, np.ndarray) and elem.ndim == 1:
        return _to_list(elem)
    else:
        return elem
