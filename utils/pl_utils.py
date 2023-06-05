import os
import sys
import gc
import glob
import datetime
import signal
from shutil import copyfile
from functools import partial
import rich
from rich.console import Console, NewLine, Segment, NO_CHANGE
import wandb
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.trainer.states import TrainerStatus


def signal_handler(trainer):
    if trainer is None:
        return
    # pylint: disable=unused-argument
    def _handler(signum, frame):
        # stop every ddp process if any world process decides to stop
        trainer.should_stop = True
        if not trainer.interrupted:
            trainer.state.status = TrainerStatus.INTERRUPTED
        pl_print(f"Interrupted {signum}: gracefully exiting...", color="red")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

def overide_print(file=None):
    import builtins
    _print = builtins.print
    if rank_zero_only.rank != 0:
        builtins.print = partial(pl_print, sink=_print)  # wrap_fn that returns None
        return
    builtins.print = Print(file=file, sink=_print)
    rich._console = RichConsole()
    rich_print = rich._console.print
    if file is not None:
        rich._console.print = partial(rich_print, file=open(file, 'a'))

@rank_zero_only
def pl_print(*args, sink=print, color="blue", **kwargs):
    color_dict = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "purple": "35"}
    note = "\033[1;" + color_dict[color] + "m♫\033[0m" if color in color_dict.keys() else ""
    sink(note, *args, **kwargs)  # show in console

@rank_zero_only
class Print(object):
    def __init__(self, file=None, sink=print):
        self.file = file
        self.sink = sink

    def __call__(self, *args, **kwargs):
        self.sink(*args, **kwargs)  # show in console
        if self.file:  # redirect to file
            if kwargs and 'file' in kwargs.keys():
                return
            with open(self.file, 'a') as f:
                self.sink(*args, file=f, **kwargs)

@rank_zero_only
class File(object):
    def __init__(self, file, sink=print):
        self._fobj = file
        self.sink = sink

    def write(self, *args, **kwargs):
        self.sink(*args, **kwargs)
        self._fobj.write(*args, **kwargs)

    def close(self, *args, **kwargs):
        self._fobj.close(*args, **kwargs)

    def __exit__(self):
        self._fobj.close()

    # def __getattr__(self, attr, *args):
    #     def g(*a, **kw):
    #         if attr == 'write':
    #             print(*a, **kw)
    #         return getattr(self._fobj, attr, *args)(*a, **kw)
    #     return g


def get_time_tag(seconds=1):
    time_tag = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
    time_tag = datetime.datetime.strftime(time_tag, "%y%m%d_%H%M%S")
    return time_tag

def to_numpy(X):
    """Generic function to convert a pytorch tensor to numpy."""
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, dict):
        return {key: to_numpy(val) for key, val in X.items()}
    if hasattr(X, 'iloc'):
        return X.values
    if isinstance(X, (tuple, list)):
        if isinstance(X[0], dict):
            return np.asarray([to_numpy(val) for val in X])
        return np.asarray(X)  # not same with np.array
    if not isinstance(X, (torch.Tensor, torch.nn.utils.rnn.PackedSequence)):
        raise TypeError("Cannot convert this data type to a numpy array.")
    if X.requires_grad:
        X = X.detach()
    if X.is_cuda:
        X = X.cpu()
    return X.numpy()

def load_state_dict(pl_module, ckpt_path):
    # if load all, use pl_module = LiteModule.load_from_checkpoint(ckpt_path, map_location)
    with pl_legacy_patch():  # avoid oom
        checkpoint = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
    if 'state_dict' in checkpoint.keys():
        pl_module.load_state_dict(checkpoint['state_dict'])
    else:
        pl_module.load_state_dict(checkpoint)
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    pl_print(f"Load ckpt in {ckpt_path}")
    return pl_module

@rank_zero_only
def save_code_wandb(root_dir, exts_saved=('.py', '.sh', '.yaml'), exclude=('outputs')):
    try:
        log_dir = wandb.run.dir
    except:
        pl_print("No WandbLogger.")
        return
    for path in glob.glob(os.path.join(root_dir, '**/*'), recursive=True):
        if log_dir in path or not path.endswith(exts_saved) or any(p in path for p in exclude):
            continue
        out_file = os.path.join(log_dir, 'code', os.path.relpath(path, root_dir))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if os.path.exists(out_file):  # train.py
            out_file = os.path.join(os.path.dirname(out_file), 'wb_' + os.path.basename(out_file))
        copyfile(path, out_file)

@rank_zero_only
class ConsoleLogger(object):
    def __init__(self, file):
        """
        sys.stdout = ConsoleLogger(os.path.join(wblogger._experiment.dir, "output.log"))
        [main codes...]
        """
        self.stdout = sys.stdout
        self.logfile = open(file, "a")

    def write(self, *args, **kwargs):
        self.stdout.write(*args, **kwargs)
        self.logfile.write(*args, **kwargs)

    def flush(self, *args, **kwargs):
        # self.stdout.flush(*args, **kwargs)
        # self.stderr.flush(*args, **kwargs)
        self.logfile.flush(*args, **kwargs)
        pass

    def isatty(self):
        # Rich: writing to a terminal and not a file.
        return True

    def __exit__(self):
        sys.stdout = self.stdout
        self.logfile.close()

class RichConsole(Console):
    def print(
        self,
        *objects,
        sep=" ",
        end="\n",
        style=None,
        justify=None,
        overflow=None,
        no_wrap=None,
        emoji=None,
        markup=None,
        highlight=None,
        width=None,
        height=None,
        crop=True,
        soft_wrap=None,
        new_line_start=False,
        file=None
    ):
        """Print to the console.

        Args:
            objects (positional args): Objects to log to the terminal.
            sep (str, optional): String to write between print data. Defaults to " ".
            end (str, optional): String to write at end of print data. Defaults to "\\\\n".
            style (Union[str, Style], optional): A style to apply to output. Defaults to None.
            justify (str, optional): Justify method: "default", "left", "right", "center", or "full". Defaults to ``None``.
            overflow (str, optional): Overflow method: "ignore", "crop", "fold", or "ellipsis". Defaults to None.
            no_wrap (Optional[bool], optional): Disable word wrapping. Defaults to None.
            emoji (Optional[bool], optional): Enable emoji code, or ``None`` to use console default. Defaults to ``None``.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use console default. Defaults to ``None``.
            highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use console default. Defaults to ``None``.
            width (Optional[int], optional): Width of output, or ``None`` to auto-detect. Defaults to ``None``.
            crop (Optional[bool], optional): Crop output to width of terminal. Defaults to True.
            soft_wrap (bool, optional): Enable soft wrap mode which disables word wrapping and cropping of text or ``None`` for
                Console default. Defaults to ``None``.
            new_line_start (bool, False): Insert a new line at the start if the output contains more than one line. Defaults to ``False``.
        """
        if not objects:
            objects = (NewLine(),)

        if soft_wrap is None:
            soft_wrap = self.soft_wrap
        if soft_wrap:
            if no_wrap is None:
                no_wrap = True
            if overflow is None:
                overflow = "ignore"
            crop = False
        render_hooks = self._render_hooks[:]
        with self:
            renderables = self._collect_renderables(
                objects,
                sep,
                end,
                justify=justify,
                emoji=emoji,
                markup=markup,
                highlight=highlight,
            )
            for hook in render_hooks:
                renderables = hook.process_renderables(renderables)
            render_options = self.options.update(
                justify=justify,
                overflow=overflow,
                width=min(width, self.width) if width is not None else NO_CHANGE,
                height=height,
                no_wrap=no_wrap,
                markup=markup,
                highlight=highlight,
            )

            new_segments = []
            extend = new_segments.extend
            render = self.render
            if style is None:
                for renderable in renderables:
                    extend(render(renderable, render_options))
            else:
                for renderable in renderables:
                    extend(
                        Segment.apply_style(
                            render(renderable, render_options), self.get_style(style)
                        )
                    )
            if new_line_start:
                if (
                    len("".join(segment.text for segment in new_segments).splitlines())
                    > 1
                ):
                    new_segments.insert(0, Segment.line())
            if crop:
                buffer_extend = self._buffer.extend
                for line in Segment.split_and_crop_lines(
                    new_segments, self.width, pad=False
                ):
                    buffer_extend(line)
            else:
                self._buffer.extend(new_segments)

            # added
            if file is not None:
                text = self._render_buffer(self._buffer[:])
                file.write(text)
            # if self._buffer_index == 0:
            #     file.flush()
