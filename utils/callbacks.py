import time
from datetime import timedelta
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from utils.pl_utils import pl_print


class Timer(Callback):
    """The Timer callback tracks the time spent in the training, validation, and test loops and interrupts the
    Trainer if the given time limit for the training loop is reached.
    """

    def __init__(self, duration=None, interval="epoch", verbose=True):
        super().__init__()
        if isinstance(duration, str):  # DD:HH:MM:SS
            dhms = duration.strip().split(":")
            dhms = [int(i) for i in dhms]
            duration = timedelta(days=dhms[0], hours=dhms[1], minutes=dhms[2], seconds=dhms[3])
        if isinstance(duration, dict):
            duration = timedelta(**duration)
        self._duration = duration.total_seconds() if duration is not None else None
        self._interval = interval
        self._verbose = verbose
        self._start_time = {stage: None for stage in RunningStage}
        self._end_time = {stage: None for stage in RunningStage}
        self._offset = 0

    def start_time(self, stage=RunningStage.TRAINING):
        """Return the start time of a particular stage (in seconds)"""
        stage = RunningStage(stage)
        return self._start_time[stage]

    def end_time(self, stage=RunningStage.TRAINING):
        """Return the end time of a particular stage (in seconds)"""
        stage = RunningStage(stage)
        return self._end_time[stage]

    def time_elapsed(self, stage=RunningStage.TRAINING):
        """Return the time elapsed for a particular stage (in seconds)"""
        start = self.start_time(stage)
        end = self.end_time(stage)
        offset = self._offset if stage == RunningStage.TRAINING else 0
        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset
        return end - start + offset

    def time_remaining(self, stage=RunningStage.TRAINING):
        """Return the time remaining for a particular stage (in seconds)"""
        if self._duration is not None:
            return self._duration - self.time_elapsed(stage)

    def on_train_start(self, trainer, pl_module):
        self._check_should_stop(trainer)
        self._start_time[RunningStage.TRAINING] = time.monotonic()

    def on_train_end(self, trainer, pl_module):
        self._check_should_stop(trainer)
        self._end_time[RunningStage.TRAINING] = time.monotonic()

    def on_validation_start(self, trainer, pl_module):
        self._check_should_stop(trainer)
        self._start_time[RunningStage.VALIDATING] = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        self._check_should_stop(trainer)
        self._end_time[RunningStage.VALIDATING] = time.monotonic()

    def on_test_start(self, trainer, pl_module):
        self._check_should_stop(trainer)
        self._start_time[RunningStage.TESTING] = time.monotonic()

    def on_test_end(self, trainer, pl_module):
        self._check_should_stop(trainer)
        self._end_time[RunningStage.TESTING] = time.monotonic()

    def on_fit_start(self, trainer, *args, **kwargs):
        # this checks the time after the state is reloaded, regardless of the interval.
        # this is necessary in case we load a state whose timer is already depleted
        self._check_should_stop(trainer)
        if self._duration is None:
            return
        self._check_time_remaining(trainer)

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self._check_should_stop(trainer)
        if self._interval != "step" or self._duration is None:
            return
        self._check_time_remaining(trainer)

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        self._check_should_stop(trainer)
        if self._interval != "epoch" or self._duration is None:
            return
        self._check_time_remaining(trainer)

    def state_dict(self):
        return {"time_elapsed": {stage.value: self.time_elapsed(stage) for stage in list(RunningStage)}}

    def load_state_dict(self, state_dict):
        time_elapsed = state_dict.get("time_elapsed", {})
        self._offset = time_elapsed.get(RunningStage.TRAINING.value, 0)

    def _check_time_remaining(self, trainer):
        assert self._duration is not None
        should_stop = self.time_elapsed() >= self._duration
        should_stop = trainer.strategy.broadcast(should_stop)  # local -> global rank
        self._check_should_stop(trainer)
        if should_stop and self._verbose:
            elapsed = timedelta(seconds=int(self.time_elapsed(RunningStage.TRAINING)))
            pl_print(f"Time limit reached. Elapsed time is {elapsed}. Signaling Trainer to stop.", color="red")

    def _check_should_stop(self, trainer):
        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(trainer.should_stop)  # global -> local rank
        trainer.should_stop = trainer.should_stop or should_stop


class plReduceLROnPlateau(ReduceLROnPlateau):
    """ Modify print function only. """

    def _reduce_lr(self, epoch):
        sink_lst = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                    new_lr_str = np.format_float_scientific(new_lr, exp_digits=1, precision=1, trim='0')
                    sink_lst.append([epoch_str, i, new_lr_str])
        if self.verbose:
            sink_lst = np.array(sink_lst)
            pl_print(f"Epoch {sink_lst[0, 0]}: reducing lr of group",
                     f"({', '.join(sink_lst[:, 1])}) to ({', '.join(sink_lst[:, 2])}).")
