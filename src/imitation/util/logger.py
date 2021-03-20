import contextlib
import os
from typing import ContextManager, Optional, Sequence

import stable_baselines3.common.logger as sb_logger

from imitation.data import types


def _build_output_formats(
    folder: types.AnyPath,
    format_strs: Sequence[str] = None,
) -> Sequence[sb_logger.KVWriter]:
    """Build output formats for initializing a Stable Baselines Logger.

    Args:
      folder: Path to directory that logs are written to.
      format_strs: An list of output format strings. For details on available
        output formats see `stable_baselines3.logger.make_output_format`.
    """
    os.makedirs(folder, exist_ok=True)
    output_formats = [sb_logger.make_output_format(f, folder) for f in format_strs]
    return output_formats


class _HierarchicalLogger(sb_logger.Logger):
    def __init__(
        self,
        default_logger: sb_logger.Logger,
        format_strs: Sequence[str] = ("stdout", "log", "csv"),
    ):
        """A logger with a context for accumulating mean values.

        Args:
          default_logger: The default logger when not in the a `accumulate_means`
            context. Also the logger to which mean values are written to when
            contexts are over.
          format_strs: An list of output format strings that should be used by
            every Logger initialized by this class during an `AccumulatingMeans`
            context. For details on available output formats see
            `stable_baselines3.logger.make_output_format`.
        """
        self.default_logger = default_logger
        self.current_logger = None
        self._cached_loggers = {}
        self._subdir = None
        self.format_strs = format_strs
        super().__init__(folder=self.default_logger.dir, output_formats=None)

    @contextlib.contextmanager
    def accumulate_means(self, subdir: types.AnyPath):
        """Temporarily modifies this _HierarchicalLogger to accumulate means values.

        During this context, `self.record(key, value)` writes the "raw" values in
        "{self.default_logger.log_dir}/{subdir}" under the key "raw/{subdir}/{key}".
        At the same time, any call to `self.record` will also accumulate mean values
        on the default logger by calling
        `self.default_logger.record_mean(f"mean/{subdir}/{key}", value)`.

        During the context, `self.record(key, value)` will write the "raw" values in
        `"{self.default_logger.log_dir}/subdir"` under the key "raw/{subdir}/key".

        After the context exits, calling `self.dump()` will write the means
        of all the "raw" values accumulated during this context to
        `self.default_logger` under keys with the prefix `mean/{subdir}/`

        Note that the behavior of other logging methods, `log` and `record_mean`
        are unmodified and will go straight to the default logger.

        Args:
          subdir: A string key which determines the `folder` where raw data is
            written and temporary logging prefixes for raw and mean data. Entering
            an `accumulate_means` context in the future with the same `subdir`
            will safely append to logs written in this folder rather than
            overwrite.
        """
        if self.current_logger is not None:
            raise RuntimeError("Nested `accumulate_means` context")

        if subdir in self._cached_loggers:
            logger = self._cached_loggers[subdir]
        else:
            folder = os.path.join(self.default_logger.dir, "raw", subdir)
            os.makedirs(folder, exist_ok=True)
            output_formats = _build_output_formats(folder, self.format_strs)
            logger = sb_logger.Logger(folder, output_formats)
            self._cached_loggers[subdir] = logger

        try:
            self.current_logger = logger
            self._subdir = subdir
            yield
        finally:
            self.current_logger = None
            self._subdir = None

    def record(self, key, val, exclude=None):
        if self.current_logger is not None:  # In accumulate_means context.
            assert self._subdir is not None
            raw_key = os.path.join("raw", self._subdir, key)
            self.current_logger.record(raw_key, val, exclude)

            mean_key = os.path.join("mean", self._subdir, key)
            self.default_logger.record_mean(mean_key, val, exclude)
        else:  # Not in accumulate_means context.
            self.default_logger.record(key, val, exclude)

    @property
    def _logger(self):
        if self.current_logger is not None:
            return self.current_logger
        else:
            return self.default_logger

    def dump(self, step=0):
        self._logger.dump(step)

    def get_dir(self) -> str:
        return self._logger.get_dir()

    def log(self, *args, **kwargs):
        self.default_logger.log(*args, **kwargs)

    def record_mean(self, key, val, exclude=None):
        self.default_logger.record_mean(key, val, exclude)

    def close(self):
        raise NotImplementedError


def _sb_logger_configure_replacement(*args, **kwargs):
    raise RuntimeError(
        "Shouldn't call stable_baselines3.logger.configure "
        "once imitation.logger.configure() has been called"
    )


def _sb_logger_reset_replacement():
    raise RuntimeError(
        "Shouldn't call stable_baselines3.logger.reset "
        "once imitation.logger.configure() has been called"
    )


def is_configured() -> bool:
    """Return True if the custom logger is active."""
    return isinstance(sb_logger.Logger.CURRENT, _HierarchicalLogger)


def configure(
    folder: types.AnyPath, format_strs: Optional[Sequence[str]] = None
) -> None:
    """Configure Stable Baselines logger to be `accumulate_means()`-compatible.

    After this function is called, `stable_baselines3.logger.{configure,reset}()`
    are replaced with stubs that raise RuntimeError.

    Args:
        folder: Argument from `stable_baselines3.logger.configure`.
        format_strs: An list of output format strings. For details on available
          output formats see `stable_baselines3.logger.make_output_format`.
    """
    # Replace `stable_baselines3.logger` methods with erroring stubs to
    # prevent unexpected logging state from mixed logging configuration.
    sb_logger.configure = _sb_logger_configure_replacement
    sb_logger.reset = _sb_logger_reset_replacement

    if format_strs is None:
        format_strs = ["stdout", "log", "csv"]
    output_formats = _build_output_formats(folder, format_strs)
    default_logger = sb_logger.Logger(folder, output_formats)
    hier_logger = _HierarchicalLogger(default_logger, format_strs)
    sb_logger.Logger.CURRENT = hier_logger
    sb_logger.log("Logging to %s" % folder)
    assert is_configured()


def record(key, val, exclude=None) -> None:
    """Alias for `stable_baselines3.logger.record`."""
    sb_logger.record(key, val, exclude)


def dump(step=0) -> None:
    """Alias for `stable_baselines3.logger.dump`."""
    sb_logger.dump(step)


def accumulate_means(subdir_name: types.AnyPath) -> ContextManager:
    """Temporarily redirect record() to a different logger and auto-track kvmeans.

    Within this context, the original logger is swapped out for a special logger
    in directory `"{current_logging_dir}/raw/{subdir_name}"`.

    The special logger's `stable_baselines3.logger.record(key, val)`, in addition
    to tracking its own logs, also forwards the log to the original logger's
    `.record_mean()` under the key `mean/{subdir_name}/{key}`.

    After the context exits, these means can be dumped as usual using
    `stable_baselines3.logger.dump()` or `imitation.util.logger.dump()`.

    Note that the behavior of other logging methods, `log` and `record_mean`
    are unmodified and will go straight to the original logger.

    This context cannot be nested.

    Args:
      subdir_name: A string key for building the logger, as described above.

    Returns:
      A context manager.
    """
    assert is_configured()
    hier_logger = sb_logger.Logger.CURRENT  # type: _HierarchicalLogger
    return hier_logger.accumulate_means(subdir_name)
