import os
import re
import socket
from enum import Enum
from functools import partial
from binascii import hexlify, unhexlify
import random
import hydra
from omegaconf import OmegaConf, open_dict
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.utils import _safe_indexing

@rank_zero_only
def pl_print(*args, sink=print, **kwargs):
    sink(*args, **kwargs)  # show in console

def set_ncclSocket():
    os.environ['NCCL_DEBUG'] = 'WARN'
    socket.setdefaulttimeout(120)
    sk = socket.socket()
    sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

def seed_everything(seed=None, workers=True, sink=pl_print, verbose=0):
    if seed is None:
        seed = int(os.environ.get("PL_GLOBAL_SEED", str(torch.initial_seed())))
    seed = seed % 2**32 if seed.bit_length() > 32 else seed
    if verbose:
        sink(f"Seed set to {seed}")
    os.environ['PYTHONHASHSEED'] = os.environ.get("PYTHONHASHSEED", str(seed))
    os.environ["PL_GLOBAL_SEED"] = os.environ.get("PL_GLOBAL_SEED", str(seed))
    os.environ["PL_SEED_WORKERS"] = os.environ.get("PL_SEED_WORKERS", f"{int(workers)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def hydra_init_no_call(cfg):
    """ Return class name and *resolved* kwargs from a hydra config. """
    cfg = cfg.copy()
    with open_dict(cfg):
        if hasattr(cfg, '_target_'):
            class_name = hydra.utils.get_class(cfg._target_)
            cfg.pop('_target_')
        else:
            class_name = None
        return class_name, OmegaConf.to_object(cfg)  # hydra.utils.instantiate(cfg)

def encode_str(s):
    """ identical integer encoding for string """
    b = s.encode()
    return int(hexlify(b), 16) if b != b'' else 0

def decode_str(i):
    b = unhexlify('%x' % i) if i != 0 else b''
    return b.decode()

def collate_fn(samples):
    return samples

def _len(x):
    return len(x)

def _apply_to_data(data, func, unpack_dict=False):
    """Apply a function to data, trying to unpack different data
    types.

    """
    apply_ = partial(_apply_to_data, func=func, unpack_dict=unpack_dict)

    if isinstance(data, dict):
        if unpack_dict:
            return [apply_(v) for v in data.values()]
        return {k: apply_(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        try:
            # e.g.list/tuple of arrays
            return [apply_(x) for x in data]
        except TypeError:
            return func(data)

    return func(data)

def flatten(arr):
    for item in arr:
        if isinstance(item, (tuple, list, dict)):
            yield from flatten(item)
        else:
            yield item

def get_len(data):
    lens = [_apply_to_data(data, _len, unpack_dict=True)]
    lens = list(flatten(lens))
    len_set = set(lens)
    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")
    return list(len_set)[0]

def infer_predict_nonlin(net):
    """Infers the correct nonlinearity to apply for this net."""
    criterion = net.criterion_
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        return partial(torch.softmax, dim=-1)
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        def _sigmoid_then_2d(x):
            prob = torch.sigmoid(x)
            y_proba = torch.stack((1 - prob, prob), 1)
            return y_proba
        return _sigmoid_then_2d

    def _identity(x):
        return x
    return _identity

def get_map_location(target_device, fallback_device='cpu'):
    """Determine the location to map loaded data (e.g., weights)
    for a given target device (e.g. 'cuda').
    """
    if target_device is None:
        target_device = fallback_device
    map_location = torch.device(target_device)
    if map_location.type == 'cuda' and not torch.cuda.is_available():
        print(f"No CUDA devices are available. Loading on device '{fallback_device}' instead.")
        map_location = torch.device(fallback_device)
    return map_location

def to_tensor(X, device=None, dtype=None, stack=False):
    """Turn input data to torch tensor."""
    if isinstance(X, (torch.Tensor, PackedSequence)):
        return to_device(X, device=device, dtype=dtype)
    if isinstance(X, dict):
        return {key: to_tensor(val, device=device, dtype=dtype) for key, val in X.items()}
    if isinstance(X, (list, tuple)):
        if isinstance(X[0], (torch.Tensor, PackedSequence)):
            if stack:
                return torch.stack([to_device(x, device=device, dtype=dtype) for x in X], dim=0)
            return type(X)([to_device(x, device=device, dtype=dtype) for x in X])
        if isinstance(X[0], str):
            # torch not support string tensor
            return X
        if isinstance(X[0], np.ndarray):
            # Creating a tensor from a list of numpy.ndarrays is extremely slow, converting to array first.
            X = np.array(X)
    return torch.as_tensor(X, device=device, dtype=dtype)

def to_numpy(X):
    """Generic function to convert a pytorch tensor to numpy."""
    if isinstance(X, (np.ndarray, int, float)):
        return X
    if isinstance(X, dict):
        return {key: to_numpy(val) for key, val in X.items()}
    if hasattr(X, 'iloc'):
        return X.values
    if isinstance(X, (tuple, list)):
        return np.asarray(X)
    if not isinstance(X, (torch.Tensor, PackedSequence)):
        raise TypeError("Cannot convert this data type to a numpy array.")
    if X.requires_grad:
        X = X.detach()
    if X.is_cuda:
        X = X.cpu()
    return X.numpy()

def to_device(X, device=None, dtype=None):
    """Generic function to modify the device type of the tensor(s) or module."""
    if device is None or not hasattr(X, 'to'):
        return X
    if isinstance(X, dict):
        return {key: to_device(val, device=device, dtype=dtype) for key, val in X.items()}
    if isinstance(X, (tuple, list)):
        return type(X)(to_device(x, device=device, dtype=dtype) for x in X)
    if dtype is None:
        return X.to(device=device)
    return X.to(device=device, dtype=dtype)


# pylint: disable=unused-argument
def _indexing_none(data, i):
    return None


def _indexing_dict(data, i):
    return {k: v[i] for k, v in data.items()}


def _indexing_list_tuple_of_data(data, i, indexings=None):
    """Data is a list/tuple of data structures (e.g. list of numpy arrays).
    ``indexings`` are the indexing functions for each element of
    ``data``. If ``indexings`` are not given, the indexing functions
    for the individual structures have to be determined ad hoc, which
    is slower.
    """
    if not indexings:
        return [multi_indexing(x, i) for x in data]
    return [multi_indexing(x, i, indexing)
            for x, indexing in zip(data, indexings)]


def _indexing_ndframe(data, i):
    # During fit, DataFrames are converted to dict, which is why we
    # might need _indexing_dict.
    if hasattr(data, 'iloc'):
        return data.iloc[i]
    return _indexing_dict(data, i)


def _indexing_other(data, i):
    # sklearn's safe_indexing doesn't work with tuples since 0.22
    if isinstance(i, (int, np.integer, slice, tuple)):
        return data[i]
    return _safe_indexing(data, i)


def _normalize_numpy_indices(i):
    """Normalize the index in case it is a numpy integer or boolean
    array."""
    if isinstance(i, np.ndarray):
        if i.dtype == bool:
            i = tuple(j.tolist() for j in i.nonzero())
        elif i.dtype == int:
            i = i.tolist()
    return i


def multi_indexing(data, i, indexing=None):
    """Perform indexing on multiple data structures.
    Currently supported data types:
    * numpy arrays
    * torch tensors
    * pandas NDFrame
    * a dictionary of the former three
    * a list/tuple of the former three
    ``i`` can be an integer or a slice.

    Examples
    --------
    >>> multi_indexing(np.asarray([1, 2, 3]), 0)
    1
    >>> multi_indexing(np.asarray([1, 2, 3]), np.s_[:2])
    array([1, 2])
    >>> multi_indexing(torch.arange(0, 4), np.s_[1:3])
    tensor([ 1.,  2.])
    >>> multi_indexing([[1, 2, 3], [4, 5, 6]], np.s_[:2])
    [[1, 2], [4, 5]]
    >>> multi_indexing({'a': [1, 2, 3], 'b': [4, 5, 6]}, np.s_[-2:])
    {'a': [2, 3], 'b': [5, 6]}
    >>> multi_indexing(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), [1, 2])
       a  b
    1  2  5
    2  3  6
    Parameters
    ----------
    data
      Data of a type mentioned above.
    i : int or slice
      Slicing index.
    indexing : function/callable or None (default=None)
      If not None, use this function for indexing into the data. If
      None, try to automatically determine how to index data.
    """
    # in case of i being a numpy array
    i = _normalize_numpy_indices(i)

    # If we already know how to index, use that knowledge
    if indexing is not None:
        return indexing(data, i)

    # If we don't know how to index, find out and apply
    return check_indexing(data)(data, i)

def check_indexing(data):
    """Perform a check how incoming data should be indexed and return an
    appropriate indexing function with signature f(data, index).

    This is useful for determining upfront how data should be indexed
    instead of doing it repeatedly for each batch, thus saving some
    time.
    """
    if data is None:
        return _indexing_none

    if isinstance(data, dict):
        # dictionary of containers
        return _indexing_dict

    if isinstance(data, (list, tuple)):
        try:
            # list or tuple of containers
            # TODO: Is there a better way than just to try to index? This
            # is error prone (e.g. if one day list of strings are
            # possible).
            multi_indexing(data[0], 0)
            indexings = [check_indexing(x) for x in data]
            return partial(_indexing_list_tuple_of_data, indexings=indexings)
        except TypeError:
            # list or tuple of values
            return _indexing_other

    if hasattr(data, 'iloc'):
        # pandas NDFrame, will be transformed to dict
        return _indexing_ndframe

    # torch tensor, numpy ndarray, list
    return _indexing_other


def get_sklearn_scorer(scoring):
    if isinstance(scoring, str):
        if hasattr(metrics, scoring):
            scorer = getattr(metrics, scoring)
        elif hasattr(metrics, scoring + '_score'):
            scorer = getattr(metrics, scoring + '_score')
        else:
            raise ValueError(
                "%r is not a valid scoring value. "
                "Use sorted(sklearn.metrics.SCORERS.keys()) "
                "to get valid options." % scoring
            )
    else:
        scorer = scoring
    return scorer

def make_sklearn_scorer(scoring):
    """If ``scoring`` is a sklearn metric function, convert it to a
    sklearn scorer and return it. Otherwise, return ``scoring`` unchanged."""
    if callable(scoring):
        module = getattr(scoring, '__module__', None)

        # those are scoring objects returned by make_scorer starting
        # from sklearn 0.22
        scorer_names = ('_PredictScorer', '_ProbaScorer', '_ThresholdScorer')
        if (
                hasattr(module, 'startswith') and
                module.startswith('sklearn.metrics.') and
                not module.startswith('sklearn.metrics.scorer') and
                not module.startswith('sklearn.metrics.tests.') and
                not scoring.__class__.__name__ in scorer_names
        ):
            return metrics.make_scorer(scoring)
    return scoring

def match(name, group):
    name = name.lower()
    for sub in group:
        if '*' in sub or '|' in sub:
            sub = sub.replace('?', '.').replace('*', '.*')
            if re.search(sub, name) is not None:
                return True
        elif sub == name:
            return True
    return False

def filter_log_keys(keys, keys_ignored=None, keys_kept=None):
    """Filter out keys that are generally to be ignored."""
    keys_ignored = keys_ignored or ()
    keys_kept = keys_kept or ()
    for key in keys:
        if (key in keys_kept or key.startswith(keys_kept) or key.endswith(keys_kept)) or \
            not (key in keys_ignored or
                 key.startswith(('event_', 'epoch')) or
                 key.endswith(('_best', '_batch_count'))):
            yield key

def sort_log_keys(keys):
    rule = {}
    ref = ('trn', 'val', 'tst', 'lr')
    for k in keys:
        if k.lower().startswith(ref):
            rule[k] = str(ref.index(k[:3]))
        else:
            rule[k] = k
    return sorted(keys, key=lambda x: rule[x])

def rename_tensorboard_key(key):
    """Rename keys from history to keys in TensorBoard
    """
    if key.startswith('trn') or key.startswith('val'):
        key = 'Loss/' + key
    return key


class Ansi(Enum):
    """Colors in `\033[Xm` or `\033[38;2;R;G;Bm` format."""
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class BaseDataset(Dataset):
    def __init__(self, X, y=None, length=None):
        self.X = X
        self.y = y
        if length is not None:
            self.length = length
        elif y is not None:
            self.length = len(y)
        else:
            self.length = len(X)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        yi = torch.Tensor([0]) if self.y is None else self.y[i]
        return self.X[i], yi
