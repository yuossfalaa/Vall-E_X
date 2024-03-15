import socket
import argparse
import collections
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict, List, Tuple, Union

import k2
import k2.version
import lhotse
import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

Pathlike = Union[str, Path]


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_dist(
        rank=None, world_size=None, master_port=None, use_ddp_launch=False, master_addr=None
):
    """
    rank and world_size are used only if use_ddp_launch is False.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)

    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("nccl")


def cleanup_dist():
    dist.destroy_process_group()


def get_git_sha1():
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
                len(
                    subprocess.run(
                        ["git", "diff", "--shortstat"],
                        check=True,
                        stdout=subprocess.PIPE,
                    )
                    .stdout.decode()
                    .rstrip("\n")
                    .strip()
                )
                > 0
        )
        git_commit = git_commit + "-dirty" if dirty_commit else git_commit + "-clean"
    except:  # noqa
        return None

    return git_commit


def get_git_date():
    try:
        git_date = (
            subprocess.run(
                ["git", "log", "-1", "--format=%ad", "--date=local"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:  # noqa
        return None

    return git_date


def get_git_branch_name():
    try:
        git_date = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:  # noqa
        return None

    return git_date


def get_env_info() -> Dict[str, Any]:
    """Get the environment information."""
    return {
        "k2-version": k2.version.__version__,
        "k2-build-type": k2.version.__build_type__,
        "k2-with-cuda": k2.with_cuda,
        "k2-git-sha1": k2.version.__git_sha1__,
        "k2-git-date": k2.version.__git_date__,
        "lhotse-version": lhotse.__version__,
        "torch-version": str(torch.__version__),
        "torch-cuda-available": torch.cuda.is_available(),
        "torch-cuda-version": torch.version.cuda,
        "python-version": sys.version[:3],
        "icefall-git-branch": get_git_branch_name(),
        "icefall-git-sha1": get_git_sha1(),
        "icefall-git-date": get_git_date(),
        "icefall-path": str(Path(__file__).resolve().parent.parent),
        "k2-path": str(Path(k2.__file__).resolve()),
        "lhotse-path": str(Path(lhotse.__file__).resolve()),
        "hostname": socket.gethostname(),
        "IP address": socket.gethostbyname(socket.gethostname()),
    }


def register_inf_check_hooks(model: nn.Module) -> None:
    """Registering forward hook on each module, to check
    whether its output tensors is not finite.

    Args:
      model:
        the model to be analyzed.
    """

    for name, module in model.named_modules():
        if name == "":
            name = "<top-level>"

        # default param _name is a way to capture the current value of the variable "name".
        def forward_hook(_module, _input, _output, _name=name):
            if isinstance(_output, Tensor):
                if not torch.isfinite(_output.to(torch.float32).sum()):
                    raise ValueError(
                        f"The sum of {_name}.output is not finite: {_output}"
                    )
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    if isinstance(o, tuple):
                        o = o[0]
                    if not isinstance(o, Tensor):
                        continue
                    if not torch.isfinite(o.to(torch.float32).sum()):
                        raise ValueError(
                            f"The sum of {_name}.output[{i}] is not finite: {_output}"
                        )

        # default param _name is a way to capture the current value of the variable "name".
        def backward_hook(_module, _input, _output, _name=name):
            if isinstance(_output, Tensor):
                if not torch.isfinite(_output.to(torch.float32).sum()):
                    logging.warning(
                        f"The sum of {_name}.grad is not finite"  # ": {_output}"
                    )
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    if isinstance(o, tuple):
                        o = o[0]
                    if not isinstance(o, Tensor):
                        continue
                    if not torch.isfinite(o.to(torch.float32).sum()):
                        logging.warning(f"The sum of {_name}.grad[{i}] is not finite")

        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

    for name, parameter in model.named_parameters():

        def param_backward_hook(grad, _name=name):
            if not torch.isfinite(grad.to(torch.float32).sum()):
                logging.warning(f"The sum of {_name}.param_grad is not finite")

        parameter.register_hook(param_backward_hook)


def setup_logger(
        log_filename: Pathlike,
        log_level: str = "info",
        use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")


class MetricsTracker(collections.defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super(MetricsTracker, self).__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            norm_value = (
                float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            )
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
            self,
            tb_writer: SummaryWriter,
            prefix: str,
            batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)
