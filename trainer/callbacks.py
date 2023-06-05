import os
import sys
import glob
import time
import signal
import threading
from numbers import Number
from shutil import copyfile
from contextlib import suppress, contextmanager
from inspect import getfullargspec
from tabulate import tabulate
import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import check_scoring, auc, precision_recall_curve, roc_curve
from sklearn.base import BaseEstimator
from pytorch_lightning.utilities import rank_zero_only
import wandb
from .utils import to_numpy, to_device, get_sklearn_scorer, make_sklearn_scorer, Ansi,\
    filter_log_keys, sort_log_keys, rename_tensorboard_key

############################## Callback ##############################
class Callback:
    """Base class for callbacks.

    All custom callbacks should inherit from this class. The subclass
    may override any of the ``on_...`` methods. It is, however, not
    necessary to override all of them, since it's okay if they don't
    have any effect.

    Classes that inherit from this also gain the ``get_params`` and
    ``set_params`` method.

    """

    def initialize(self):
        """(Re-)Set the initial state of the callback. Use this
        e.g. if the callback tracks some state that should be reset
        when the model is re-initialized.

        This method should return self.

        """
        return self

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        """Called at the beginning of training."""

    def on_train_end(self, net, X=None, y=None, **kwargs):
        """Called at the end of training."""

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the end of each epoch."""

    def on_batch_begin(self, net, batch=None, training=None, **kwargs):
        """Called at the beginning of each batch."""

    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        """Called at the end of each batch."""

    def on_grad_computed(self, net, named_parameters, X=None, y=None, training=None, **kwargs):
        """Called once per batch after gradients have been computed but before
        an update step was performed.
        """

    def _get_param_names(self):
        return (key for key in self.__dict__ if not key.endswith('_'))

    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep=deep)

    def set_params(self, **params):
        BaseEstimator.set_params(self, **params)


############################## Signal Handling ##############################
def register_signal(signum, handlers):
    # signal.signal(signum, handlers)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signum, handlers)

class SignalConnector:
    def __init__(self, trainer):
        self.trainer = trainer
        self.trainer.should_stop_ = False
        self.trainer.signal_handling_ = False
        self.original_handlers_ = {}

    def register(self):
        if self.trainer.signal_handling_:
            return
        self.trainer.signal_handling_ = True
        # original valid handlers
        valid_signals = signal.valid_signals() - {signal.SIGKILL, signal.SIGSTOP}
        self.original_handlers_ = {signum: signal.getsignal(signum) for signum in valid_signals}

        def _handler(signum, frame):
            if signum != 15:
                self.trainer.warning(f"\033[1;31m♫{self.trainer.local_rank}\033[0m Gracefully terminating {signum}...")
            self.trainer.should_stop_ = True
        register_signal(signal.SIGINT, _handler)
        register_signal(signal.SIGTERM, _handler)

    def teardown(self):
        for signum, handler in self.original_handlers_.items():
            if handler is not None:
                register_signal(signum, handler)
        self.original_handlers_ = {}
        self.trainer.signal_handling_ = False


############################## Utils ##############################
@rank_zero_only
def wb_pr_curve(y_true=None, y_probas=None, labels=None, classes_to_plot=None, title=None, interp_size=21):
    """
    Arguments:
        y_true (arr): true sparse labels
        y_probas (arr): Target scores, can either be probability estimates, confidence
                         values, or non-thresholded measure of decisions.
                         shape: (*y_true.shape, num_classes)
        labels (list): Named labels for target variable (y). Makes plots easier to
                        read by replacing target values with corresponding index.
                        For example labels = ['dog', 'cat', 'owl'] all 0s are
                        replaced by 'dog', 1s by 'cat'.
        classes_to_plot (list): unique values of y_true to include in the plot

    Returns:
        Table of PR. To see plots, go to your W&B run page then expand the 'media' tab
        under 'auto visualizations'.
    """

    def _step(x):
        y = np.array(x)
        for i in range(1, len(y)):
            y[i] = max(y[i], y[i - 1])
        return y

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    classes = np.unique(y_true)
    if classes_to_plot is None:
        classes_to_plot = classes
    if not isinstance(classes_to_plot, list):
        classes_to_plot = [classes_to_plot]

    precision = dict()
    interp_recall = np.linspace(0, 1, interp_size)[::-1]
    indices_to_plot = np.where(np.isin(classes, classes_to_plot))[0]
    for i in indices_to_plot:
        if labels is not None and (
            isinstance(classes[i], int) or isinstance(classes[0], np.integer)
        ):
            class_label = labels[classes[i]]
        else:
            class_label = classes[i]

        cur_precision, cur_recall, _ = precision_recall_curve(
            y_true, y_probas[:, i], pos_label=classes[i]
        )
        # smooth the precision (monotonically increasing)
        cur_precision = _step(cur_precision)
        # reverse order so that recall in ascending
        cur_precision = cur_precision[::-1]
        cur_recall = cur_recall[::-1]
        indices = np.searchsorted(cur_recall, interp_recall, side="left")
        precision[class_label] = cur_precision[indices]

    df = pd.DataFrame(
        {
            "class": np.hstack([[k] * len(v) for k, v in precision.items()]),
            "precision": np.hstack(list(precision.values())),
            "recall": np.tile(interp_recall, len(precision)),
        }
    )
    auc_pr = []
    for c in classes_to_plot:
        auc_i = auc(x=df.loc[df['class'].isin([c]), 'recall'], y=df.loc[df['class'].isin([c]), 'precision'])
        auc_pr.append(f"{c}: {auc_i:.4f}")
    title = (title or "PR") + "(" + ",".join(auc_pr) + ")"
    wandb.log({title: wandb.plot_table(
        "wandb/area-under-curve/v0",
        wandb.Table(dataframe=df.round(3)),
        {"x": "recall", "y": "precision", "class": "class"},
        {"title": title},
    )})
    return df

@rank_zero_only
def wb_roc_curve(y_true=None, y_probas=None, labels=None, classes_to_plot=None, title=None):
    """
    Calculates receiver operating characteristic scores and visualizes them as the
    ROC curve.

    Arguments:
        y_true (arr): true sparse labels
        y_probas (arr): Target scores, can either be probability estimates, confidence
                         values, or non-thresholded measure of decisions.
                         shape: (*y_true.shape, num_classes)
        labels (list): Named labels for target variable (y). Makes plots easier to
                        read by replacing target values with corresponding index.
                        For example labels = ['dog', 'cat', 'owl'] all 0s are
                        replaced by 'dog', 1s by 'cat'.
        classes_to_plot (list): unique values of y_true to include in the plot

    Returns:
        Table of ROC. To see plots, go to your W&B run page then expand the 'media' tab
            under 'auto visualizations'.
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    classes = np.unique(y_true)
    if classes_to_plot is None:
        classes_to_plot = classes
    if not isinstance(classes_to_plot, list):
        classes_to_plot = [classes_to_plot]

    fpr, tpr = dict(), dict()
    indices_to_plot = np.where(np.isin(classes, classes_to_plot))[0]
    for i in indices_to_plot:
        if labels is not None and (
            isinstance(classes[i], int) or isinstance(classes[0], np.integer)
        ):
            class_label = labels[classes[i]]
        else:
            class_label = classes[i]

        fpr[class_label], tpr[class_label], _ = roc_curve(
            y_true, y_probas[..., i], pos_label=classes[i]
        )

    df = pd.DataFrame(
        {
            "class": np.hstack([[k] * len(v) for k, v in fpr.items()]),
            "fpr": np.hstack(list(fpr.values())),
            "tpr": np.hstack(list(tpr.values())),
        }
    )
    auc_roc = []
    for c in classes_to_plot:
        auc_i = auc(x=df.loc[df['class'].isin([c]), 'fpr'], y=df.loc[df['class'].isin([c]), 'tpr'])
        auc_roc.append(f"{c}: {auc_i:.4f}")
    title = (title or "ROC") + "(" + ",".join(auc_roc) + ")"
    wandb.log({title: wandb.plot_table(
        "wandb/area-under-curve/v0",
        wandb.Table(dataframe=df.round(3)),
        {"x": "fpr", "y": "tpr", "class": "class"},
        {
            "title": title,
            "x-axis-title": "False positive rate",
            "y-axis-title": "True positive rate",
        },
    )})
    return df


def format_float_list(lst):
    res = []
    for v in lst:
        v = np.format_float_scientific(v, exp_digits=1, precision=1, trim='0')
        res.append(str(v).replace('e', ''))
    return ','.join(res)


############################## Scoring ##############################

class EpochScoring(Callback):
    """Callback that performs generic scoring on predictions using `sklearn.metrics`."""

    def __init__(
            self,
            scoring,
            lower_is_better=False,
            on_train=False,
            name=None,
            rescore=False,
            lang='',
            save=False,
            **kwargs,
    ):
        self.scoring = scoring
        self.lower_is_better = lower_is_better
        self.on_train = on_train
        self.name = name
        self.rescore = rescore
        self.lang = lang
        self.save = save
        vars(self).update(kwargs)

    def set_params(self, **params):
        vars(self).update(params)

    @property
    def kwargs(self):
        """These `kwargs` are passed to scorer."""
        excluded = ['scoring', 'lower_is_better', 'on_train', 'name', 'rescore', 'lang', 'save']
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def _initialize_cache(self):
        self.y_trues_ = []
        self.y_preds_ = []
        self.spks_ = []
        self.langs_ = []
        self.best_scores_ = []

    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        self._initialize_cache()
        return self

    # pylint: disable=attribute-defined-outside-init,arguments-differ,signature-differs
    def on_train_begin(self, net, **kwargs):
        # if net.local_rank != 0:
        #     return
        # Looks for the right most index where `*_best` is True
        # That index is used to get the best score in `net.history`
        with suppress(ValueError, IndexError, KeyError):
            best_name_history = net.history[:, '{}_best'.format(self.name)]
            idx_best_reverse = best_name_history[::-1].index(True)
            idx_best = len(best_name_history) - idx_best_reverse - 1
            self.best_score_ = net.history[idx_best, self.name]

    # pylint: disable=arguments-differ,unused-argument,signature-differs
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, dataset_test=None, **kwargs):
        dataset = {True: dataset_train, False: dataset_valid, "": dataset_test}[self.on_train]
        if dataset is None:
            return
        self._initialize_cache()

    # pylint: disable=arguments-differ
    def on_batch_end(self, net, batch, training, y_pred, y_true, spk=None, lang=None, **kwargs):
        if training != self.on_train:
            return
        if y_true is not None:
            self.y_trues_.append(y_true)
        if y_pred is not None:
            self.y_preds_.append(y_pred)
        if spk is not None:
            self.spks_.append(spk)
        if lang is not None:
            self.langs_.append(lang)

    def vote(self, y_pred, y_true=None, mode='soft'):
        y_pred = np.array(y_pred)
        if mode == 'soft':
            return np.mean(y_pred, axis=0)
        if y_true is not None and y_pred.shape != y_true.shape and mode == 'hard':
            y_pred = y_pred.argmax(axis=-1)
        return np.round(np.mean(y_pred, axis=0))

    # pylint: disable=unused-argument,arguments-differ,signature-differs
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, dataset_test=None, **kwargs):
        """ Score when matchs rank, on_train and val_interval. """
        dataset = {True: dataset_train, False: dataset_valid, "": dataset_test}[self.on_train]
        if dataset is None or len(self.y_trues_) == 0 or len(self.y_preds_) == 0:
            return
        y_true = np.concatenate([to_numpy(y) for y in self.y_trues_])
        y_pred = np.concatenate([to_numpy(y) for y in self.y_preds_])
        if self.spks_:
            spks = np.concatenate([to_numpy(y) for y in self.spks_])
        if self.lang and self.langs_:
            langs = np.concatenate([to_numpy(y) for y in self.langs_])  # 0: en, 1: hk
            target_lang = {'en': 0, 'hk': 1}[self.lang]
            idx = np.isin(langs, target_lang)
            y_true, y_pred = y_true[idx], y_pred[idx]
            if self.spks_:
                spks = spks[idx]

        if self.rescore and self.spks_:  # rescoring with same speakers
            res = {}
            for i in range(len(y_true)):
                k = spks[i]
                if k not in res.keys():
                    res[k] = {'y_true': [], 'y_pred': []}
                res[k]['y_true'].append(y_true[i])
                res[k]['y_pred'].append(y_pred[i])
            y_true = np.array([np.mean(v['y_true'], axis=0) for _, v in res.items()])
            # soft vote
            y_pred = np.array([self.vote(y_pred=v['y_pred'], y_true=np.mean(v['y_true'], axis=0), mode=self.rescore)
                               for _, v in res.items()])
        scorer = get_sklearn_scorer(self.scoring)
        y_pred_ = y_pred if y_pred.shape == y_true.shape else y_pred.argmax(axis=-1)
        current_score = scorer(y_true, y_pred_, **self.kwargs)
        self._record_score(net.history, current_score)
        if self._is_best_score(current_score):
            self.best_scores_ = [y_true, y_pred]
            if net.local_rank == 0 and self.save:
                np.savez(os.path.join(net.saved_dir, self.name + '_best' + '.npz'), y_true=y_true, y_pred=y_pred)

    def on_train_end(self, net, **kwargs):
        if net.local_rank == 0 and self.save and self.y_trues_ and self.y_preds_:  # last scores
            y_true = np.concatenate([to_numpy(y) for y in self.y_trues_])
            y_pred = np.concatenate([to_numpy(y) for y in self.y_preds_])
            np.savez(os.path.join(net.saved_dir, self.name + '_last' + '.npz'), y_true=y_true, y_pred=y_pred)
        if net.local_rank == 0 and self.best_scores_ and wandb.run is not None:
            y_true, y_pred = self.best_scores_  # y_pred should be probas with shape [*y_true.shape, n_classes]
            # FIXME: too many indices for array: array is 1-dimensional, but 2 were indexed
            try:
                wb_pr_curve(y_true, y_pred, classes_to_plot=1)  # plot ad class
                wb_roc_curve(y_true, y_pred, classes_to_plot=1)  # plot ad class
            except:
                pass
        self._initialize_cache()

    def _record_score(self, history, current_score):
        """Record the current store and, if applicable, if it's the best score yet."""
        history.record(self.name, current_score)
        is_best = self._is_best_score(current_score)
        if is_best is None:
            return
        history.record(self.name + '_best', bool(is_best))
        if is_best:
            self.best_score_ = current_score

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score <= self.best_score_
        return current_score >= self.best_score_


class PassthroughScoring(Callback):
    """Creates scores on epoch level based on batch level scores."""

    def __init__(
            self,
            name,
            lower_is_better=True,
    ):
        self.name = name
        self.lower_is_better = lower_is_better

    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        return self

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score <= self.best_score_
        return current_score >= self.best_score_

    def get_avg_score(self, history):
        bs_key = self.name.split('_')[0] + '_batch_size'
        weights, scores = list(zip(*history[-1, 'batches', :, [bs_key, self.name]]))
        score_avg = np.average(scores, weights=weights)
        return score_avg

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self, net, **kwargs):
        history = net.history
        try:  # don't raise if there is no valid data
            history[-1, 'batches', :, self.name]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name, score_avg)
        if is_best is not None:
            history.record(self.name + '_best', bool(is_best))


############################## Logging ##############################
class EpochTimer(Callback):
    """Measures the duration of each epoch and writes it to the
    history with the name ``dur``.
    """

    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)
        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = time.time()

    def on_epoch_end(self, net, **kwargs):
        net.history.record('dur', time.time() - self.epoch_start_time_)


class LearningRate(Callback):
    def on_epoch_end(self, net, **kwargs):
        current_lrs = list(set([group['lr'] for group in net.optimizer_.param_groups]))
        net.history.record('lr', current_lrs)


class PrintLog(Callback):
    """Print useful information from the model's history as a table.
    keys_ignored: str or list of str (default=None). starting with 'event_' or ending
        on '_best' are ignored by default.
    tablefmt: 'simple', 'plain', 'grid', 'pipe', 'html', 'latex' etc. in ``tabulate``.
    floatfmt : str (default='.4f')
    stralign : 'right', 'left', 'center', or ``None``
    """

    def __init__(
            self,
            keys_ignored=None,
            group_colors=True,
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
            stralign='right',
    ):
        self.keys_ignored = keys_ignored
        self.group_colors = group_colors
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign

    def initialize(self):
        self.show_headers_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable).

        """
        value = row[key]
        if isinstance(value, bool) or value is None:
            return '+' if value else ''
        if 'lr' in key:
            return format_float_list(value)
        if not isinstance(value, Number):
            return value
        if 'dur' in key:
            return int(value)
        if 'epoch' in key:
            return int(value) - 1
        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)

    def _sorted_keys(self, keys):
        """Sort keys, dropping the ones that should be ignored.

        The keys that are in ``self.ignored_keys`` or that end on
        '_best' are dropped. Among the remaining keys:
          * 'epoch' is put first;
          * 'dur' is put last;
          * keys that start with 'event_' are put just before 'dur';
          * all remaining keys are sorted alphabetically.
        """
        sorted_keys = []

        # make sure 'epoch' comes first
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored_):
            sorted_keys.append('epoch')

        # ignore keys like *_best or event_*
        for key in filter_log_keys(sort_log_keys(keys), keys_ignored=self.keys_ignored_):
            if key != 'dur':
                sorted_keys.append(key)

        # add event_* keys
        for key in sorted(keys):
            if key.startswith('event_') and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        # make sure 'dur' comes last
        if ('dur' in keys) and ('dur' not in self.keys_ignored_):
            sorted_keys.append('dur')

        return sorted_keys

    def table(self, row):
        formatted_dict = {}
        colors = [color.value for color in Ansi if color != color.ENDC]
        if not hasattr(self, 'group'):
            self.key_colors = {}
        for idx, key in enumerate(self._sorted_keys(row.keys())):
            color_key = key.split('_')[0]
            color_key = color_key[:3] if color_key.startswith(('trn', 'val', 'tst')) else color_key
            if color_key not in self.key_colors.keys():
                self.key_colors[color_key] = colors[len(self.key_colors.keys()) % len(colors)]
            color = self.key_colors[color_key] if self.group_colors else colors[idx % len(colors)]
            formatted = self.format_row(row, key, color=color)
            if key.startswith('event_'):
                key = key[6:]
            formatted_dict[key] = formatted

        if not hasattr(self, 'headers') or not set(formatted_dict.keys()).issubset(self.headers):
            self.headers = formatted_dict.keys()
            self.show_headers_ = True
        formatted = [formatted_dict[k] if k in formatted_dict.keys() else '' for k in self.headers]
        return tabulate(
            [formatted],
            headers=self.headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
            stralign=self.stralign,
        )

    def _sink(self, text, verbose):
        if (self.sink not in self.default_sink) or verbose:
            self.sink(text)

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        self.default_sink = [print, net.print]
        data = net.history[-1]
        verbose = net.verbose
        tabulated = self.table(data)

        if self.show_headers_:
            header, lines = tabulated.split('\n', 2)[:2]
            self._sink(header, verbose)
            self._sink(lines, verbose)
            self.show_headers_ = False

        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)
        if self.sink in self.default_sink:
            sys.stdout.flush()


class TensorBoard(Callback):
    """Logs results from history to TensorBoard
    """

    def __init__(
            self,
            writer,
            close_after_train=True,
            keys_ignored=None,
            key_mapper=rename_tensorboard_key,
    ):
        self.writer = writer
        self.close_after_train = close_after_train
        self.keys_ignored = keys_ignored
        self.key_mapper = key_mapper

    def initialize(self):
        self.first_batch_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def on_batch_end(self, net, **kwargs):
        self.first_batch_ = False

    def add_scalar_maybe(self, history, key, tag, global_step=None):
        """Add a scalar value from the history to TensorBoard
        """
        hist = history[-1]
        val = hist.get(key)
        if val is None:
            return

        global_step = global_step if global_step is not None else hist['epoch']
        with suppress(NotImplementedError):
            # pytorch raises NotImplementedError on wrong types
            self.writer.add_scalar(
                tag=tag,
                scalar_value=val,
                global_step=global_step,
            )

    def on_epoch_end(self, net, **kwargs):
        """Automatically log values from the last history step."""
        history = net.history
        hist = history[-1]
        epoch = hist['epoch']

        for key in filter_log_keys(hist, keys_ignored=self.keys_ignored_):
            tag = self.key_mapper(key)
            self.add_scalar_maybe(history, key=key, tag=tag, global_step=epoch)

    def on_train_end(self, net, **kwargs):
        if self.close_after_train:
            self.writer.close()


class WandbLogger(Callback):
    """Logs model and metrics to `Weights & Biases <https://docs.wandb.com/>`.
    """

    def __init__(
            self,
            mointor='val_loss_best',
            save_model=[],
            sync_final=False,
            keys_ignored=None,
            exts_saved=('.py', '.sh', '.yaml'),
            **kwargs
    ):
        self.mointor = mointor
        self.save_model = save_model if isinstance(save_model, list) else [save_model]
        self.sync_final = sync_final
        self.keys_ignored = keys_ignored
        self.exts_saved = exts_saved
        vars(self).update(kwargs)

    def set_params(self, **params):
        vars(self).update(params)

    @property
    def kwargs(self):
        """These `kwargs` are passed to scorer."""
        excluded = ['mointor', 'save_model', 'sync_final', 'keys_ignored', 'exts_saved']
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def initialize(self):
        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def wandb_init(self):
        wandb_kwargs = {'reinit': True, 'mode': 'offline', 'save_code': True} | self.kwargs
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if k in getfullargspec(wandb.init).args}
        self.wandb_run_ = wandb.init(**wandb_kwargs)
        if wandb_kwargs['save_code']:
            log_dir = os.path.relpath(self.wandb_run_.dir).split(os.sep)[0]
            for path in glob.glob('**/*', recursive=True):
                if log_dir in path or not path.endswith(self.exts_saved):
                    continue
                out_file = os.path.join(self.wandb_run_.dir, 'code', path)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                if os.path.exists(out_file):
                    out_file = os.path.join(os.path.dirname(out_file), 'wb_' + os.path.basename(out_file))
                copyfile(path, out_file)

    def on_train_begin(self, net, **kwargs):
        """Log model topology and add a hook for gradients"""
        if net.local_rank != 0:
            return
        self.wandb_init()
        setattr(net, 'saved_dir', self.wandb_run_.dir)
        self.wandb_run_.config.wandb_dir = self.wandb_run_.dir
        net.print('\033[1;34mwandb dir\033[0m: ', self.wandb_run_.dir)
        self.wandb_run_.watch(net.unwrap(net.module_))

    def save_net(self, net, prefix='best', check=None):
        model_file = os.path.join(self.wandb_run_.dir, prefix + '_model.pt')
        net.save_params(f_module=model_file)
        if net.history[-1]['epoch'] == 0 and check:
            net.load_params(f_module=model_file)  # check saved model

    def on_epoch_end(self, net, **kwargs):
        """Log values from the last history step and save best model"""
        if net.local_rank != 0:
            return
        hist = net.history[-1]
        keys_kept = filter_log_keys(hist, keys_ignored=self.keys_ignored_)
        logged_vals = {k: hist[k] for k in keys_kept} | {'epoch': hist['epoch'] - 1, 'lr': hist['lr'][0]}
        self.wandb_run_.log(logged_vals, step=hist['epoch'] - 1)
        # save model
        self.last_vals = logged_vals
        if 'epoch' in self.save_model:
            self.save_net(net, prefix=f"epoch{hist['epoch'] - 1}")
        if self.mointor not in hist.keys():
            return
        if hist[self.mointor]:
            self.best_vals = logged_vals
            if 'best' in self.save_model:
                self.save_net(net, prefix='best')

    def on_train_end(self, net, **kwargs):
        if net.local_rank != 0:
            return
        if 'last' in self.save_model:
            self.save_net(net, prefix='last')
        self.saved_vals = self.saved_vals if hasattr(self, 'saved_vals') else {}
        if hasattr(self, 'last_vals'):
            self.saved_vals = self.saved_vals | {'last_' + k: v for k, v in self.last_vals.items()}
        if hasattr(self, 'best_vals'):
            for k, v in self.best_vals.items():
                self.saved_vals['best_' + k] = v
                self.wandb_run_.summary[k] = v
        # plots (pr, roc_curve is ploted in EpochScoring callback.)
        self.log_step_loss(net)
        # finish
        model = net.unwrap(net.module_)
        if hasattr(model, "_wandb_hook_names"):
            self.wandb_run_.unwatch(model)
        self.wandb_run_.finish()
        wandb.finish()
        if self.sync_final and self.wandb_run_._hooks.exit_code == 0 and self.kwargs.get('mode') != 'online' \
                and not net.should_stop_ and len(net.history) > 5:
            # exit_code: 0 (success), 1 (fail), 255 (interrupted)
            os.system(f"wandb sync {os.path.dirname(self.wandb_run_.dir)} &")

    def log_step_loss(self, net):
        # net.history is a list (len=epoch) of dict ('batches', metrics, 'lr', ...),
        # hist[epoch]['batches'] is a list (len=step) of dict ('*_loss', '*_batch_size')
        step_losses = {'trn_loss': [], 'val_loss': [], 'tst_loss': []}
        for ep in range(len(net.history)):
            for step in net.history[ep]['batches']:  # list of dict ('*_loss', '*_batch_size')
                for k, v in step.items():
                    if k in step_losses.keys():
                        step_losses[k].append(v)
        for k, v in step_losses.items():
            if not v:
                return
            self.plot_line(np.arange(len(v)), v, title=k + '_step', columns=['step', 'loss'])

    def plot_line(self, x_values, y_values, title, columns=["x", "y"]):
        data = [[x, y] for (x, y) in zip(x_values, y_values)]
        table = wandb.Table(data=data, columns=columns)
        self.wandb_run_.log({title: wandb.plot.line(table, columns[0], columns[1], title=title)})

    def plot_pr(self, y_trues, y_preds, title, labels=None, classes_to_plot=None):
        self.wandb_run_.log({title: wandb.plot.pr_curve(
            y_trues, y_preds, labels=labels, classes_to_plot=classes_to_plot)})

    def plot_roc(self, y_trues, y_preds, title, labels=None, classes_to_plot=None):
        self.wandb_run_.log({title: wandb.plot.roc_curve(
            y_trues, y_preds, labels=labels, classes_to_plot=classes_to_plot)})

    def plot_hist(self, x_values, y_values, title, columns=["class", "prob"]):
        data = [[x, y] for (x, y) in zip(x_values, y_values)]
        table = wandb.Table(data=data, columns=columns)  # e.g. per class probs
        self.wandb_run_.log({title: wandb.plot.histogram(table, value=columns[1], title=title)})

############################## Scheduling ##############################
def _check_lr(name, optimizer, lr):
    """Return one learning rate for each param group."""
    n = len(optimizer.param_groups)
    if not isinstance(lr, (list, tuple)):
        return lr * np.ones(n)

    if len(lr) != n:
        raise ValueError("{} lr values were passed for {} but there are "
                         "{} param groups.".format(n, name, len(lr)))
    return np.array(lr)


class LRScheduler(Callback):
    """Callback that sets the learning rate of each
    parameter group according to some policy.

    Parameters
    ----------

    policy : str or _LRScheduler class (default='WarmRestartLR')
      Learning rate policy name or scheduler to be used.

    monitor : str or callable (default=None)
      Value of the history to monitor or function/callable. In
      the latter case, the callable receives the net instance as
      argument and is expected to return the score (float) used to
      determine the learning rate adjustment.

    event_name: str, (default='event_lr')
      Name of event to be placed in history when the scheduler takes a step.
      Pass ``None`` to disable placing events in history.
      **Note:** This feature works only for pytorch version >=1.4

    step_every: str, (default='epoch')
      Value for when to apply the learning scheduler step. Can be either 'batch'
       or 'epoch'.

    kwargs
      Additional arguments passed to the lr scheduler.

    """

    def __init__(self,
                 policy='WarmRestartLR',
                 monitor='val_loss',
                 event_name="event_lr",
                 step_every='epoch',
                 **kwargs):
        self.policy = policy
        self.monitor = monitor
        self.event_name = event_name
        self.step_every = step_every
        vars(self).update(kwargs)

    def simulate(self, steps, initial_lr):
        """
        Simulates the learning rate scheduler.

        Parameters
        ----------
        steps: int
          Number of steps to simulate

        initial_lr: float
          Initial learning rate

        Returns
        -------
        lrs: numpy ndarray
          Simulated learning rates

        """
        test = torch.ones(1, requires_grad=True)
        opt = torch.optim.SGD([{'params': test, 'lr': initial_lr}])
        policy_cls = self._get_policy_cls()
        sch = policy_cls(opt, **self.kwargs)

        lrs = []
        for _ in range(steps):
            opt.step()  # suppress warning about .step call order
            lrs.append(opt.param_groups[0]['lr'])
            sch.step()

        return np.array(lrs)

    def initialize(self):
        self.policy_ = self._get_policy_cls()
        self.lr_scheduler_ = None
        self.batch_idx_ = 0
        return self

    def _get_policy_cls(self):
        if isinstance(self.policy, str):
            return getattr(sys.modules[__name__], self.policy)
        return self.policy

    @property
    def kwargs(self):
        # These are the parameters that are passed to the
        # scheduler. Parameters that don't belong there must be
        # excluded.
        excluded = ('policy', 'monitor', 'event_name', 'step_every')
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def on_train_begin(self, net, **kwargs):
        if net.history:
            try:
                self.batch_idx_ = sum(net.history[:, 'trn_batch_count'])
            except KeyError:
                self.batch_idx_ = sum(len(b) for b in net.history[:, 'batches'])
        self.lr_scheduler_ = self._get_scheduler(
            net, self.policy_, **self.kwargs
        )

    def on_epoch_end(self, net, **kwargs):
        if self.step_every != 'epoch':
            return
        if isinstance(self.lr_scheduler_, ReduceLROnPlateau):
            if callable(self.monitor):
                score = self.monitor(net)
            else:
                try:
                    score = net.history[-1, self.monitor]
                except KeyError as e:
                    raise ValueError(
                        f"'{self.monitor}' was not found in history. A "
                        f"Scoring callback with name='{self.monitor}' "
                        "should be placed before the LRScheduler callback"
                    ) from e

            self.lr_scheduler_.step(score)
            # ReduceLROnPlateau does not expose the current lr so it can't be recorded
        else:
            if self.event_name is not None and hasattr(
                    self.lr_scheduler_, "get_last_lr"):
                net.history.record(self.event_name,
                                   self.lr_scheduler_.get_last_lr()[0])
            self.lr_scheduler_.step()

    def on_batch_end(self, net, training, **kwargs):
        if not training or self.step_every != 'batch':
            return
        if self.event_name is not None and hasattr(
                self.lr_scheduler_, "get_last_lr"):
            net.history.record_batch(self.event_name,
                                     self.lr_scheduler_.get_last_lr()[0])
        if (self.batch_idx_ + 1) % net.accumulate_steps == 0:
            self.lr_scheduler_.step()
        self.batch_idx_ += 1

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
        """Return scheduler, based on indicated policy, with appropriate
        parameters.
        """
        if policy not in [ReduceLROnPlateau] and \
                'last_epoch' not in scheduler_kwargs:
            last_epoch = len(net.history) - 1
            scheduler_kwargs['last_epoch'] = last_epoch

        return policy(net.optimizer_, **scheduler_kwargs)


class WarmRestartLR(_LRScheduler):
    """Stochastic Gradient Descent with Warm Restarts (SGDR) scheduler."""

    def __init__(
            self, optimizer,
            min_lr=1e-6,
            max_lr=0.05,
            base_period=10,
            period_mult=2,
            last_epoch=-1
    ):
        self.min_lr = _check_lr('min_lr', optimizer, min_lr)
        self.max_lr = _check_lr('max_lr', optimizer, max_lr)
        self.base_period = base_period
        self.period_mult = period_mult
        super(WarmRestartLR, self).__init__(optimizer, last_epoch)

    def _get_current_lr(self, min_lr, max_lr, period, epoch):
        return min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(epoch * np.pi / period))

    def get_lr(self):
        epoch_idx = float(self.last_epoch)
        current_period = float(self.base_period)
        while epoch_idx / current_period > 1.0:
            epoch_idx -= current_period + 1
            current_period *= self.period_mult

        current_lrs = self._get_current_lr(
            self.min_lr,
            self.max_lr,
            current_period,
            epoch_idx
        )
        return current_lrs.tolist()


class EarlyStopping(Callback):
    """Callback for stopping training when scores don't improve.

    Stop training early if a specified `monitor` metric did not
    improve in `patience` number of epochs by at least `threshold`.
    """

    def __init__(
            self,
            monitor='val_loss',
            patience=-1,
            min_epochs=0,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
            save_last=False,
    ):
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.min_epochs = min_epochs
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.misses_ = 0
        self.dynamic_threshold_ = None
        self.save_last = save_last

    # pylint: disable=arguments-differ
    def on_train_begin(self, net, **kwargs):
        if self.threshold_mode not in ['rel', 'abs']:
            raise ValueError("Invalid threshold mode: '{}'".format(self.threshold_mode))
        self.misses_ = 0
        self.dynamic_threshold_ = np.inf if self.lower_is_better else -np.inf

    def epoch_end_fn(self, net, **kwargs):
        return

    def on_epoch_end(self, net, **kwargs):
        if self.patience < 0:
            return
        try:
            current_score = net.history[-1, self.monitor]
        except:
            # monitor was not found in history.
            # net.check_stop(save=self.save_last)
            print(f"Monitor {self.monitor} not found in rank {net.local_rank}!")
            setattr(net, 'should_stop_', True)
        net.check_stop(save=self.save_last)
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
        self.epoch_end_fn(net, **kwargs)
        if self.misses_ >= self.patience and len(net.history) >= self.min_epochs:
            net.print(f"Stopped as {self.monitor} has not been {'decreased' if self.lower_is_better else 'improved'}",
                      f"in the last {self.patience} epochs.")
            setattr(net, 'should_stop_', True)
        net.check_stop(save=self.save_last)

    def _is_score_improved(self, score):
        if self.lower_is_better:
            return score <= self.dynamic_threshold_
        return score >= self.dynamic_threshold_

    def _calc_new_threshold(self, score):
        """Determine threshold based on score."""
        if self.threshold_mode == 'rel':
            abs_threshold_change = self.threshold * score
        else:
            abs_threshold_change = self.threshold

        if self.lower_is_better:
            new_threshold = score - abs_threshold_change
        else:
            new_threshold = score + abs_threshold_change
        return new_threshold
