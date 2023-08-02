"""QuantumMASALA's logging module.

This module manages logging of events/messages generated across all the MPI
processes when running QuantumMASALA. The module also provides timers and counters
to track/profile routines across the package.

Notes
-----
The timers are automatically disabled when running in interactive mode to
prevent exceptions due to improper timer usage like, for example,
stopping an already stopped timer, accessing a non-existing timer, etc.
"""
# from __future__ import annotations
from qtm.typing import Optional
__all__ = ['LOGGER_NAME', 'LOG_FORMAT', 'qtmlogger',
           'qtmlogger_set_filehandle',
           'QTMTimer', 'QTMCounter', 'QTMLogger', 'warn',
           ]

import logging
from pathlib import Path
import warnings
from time import perf_counter, strftime
from importlib.util import find_spec

import numpy as np

LOGGER_NAME: str = 'QTM'
"""Global name of logger"""
LOG_FORMAT: str = "%(asctime)s - proc {MPI_WORLD_RANK}/{MPI_WORLD_SIZE} - " \
                  "level %(levelno)s: %(message)s"
"""Basic Format for log messages:
"time - proc #rank/#size - level : #level: message"
"""

TIMER_TYPE: np.dtype = np.dtype([('label', 'U30'), ('call', 'i8'), ('time', 'f8'),
                                 ('status', 'b'), ('start_time', 'f8')])
"""Datatype of the NumPy Structured array used for storing timers in `QTMTimer`
"""
COUNTER_TYPE: np.dtype = np.dtype([('label', 'U30'), ('count', 'i8')])
"""Datatype of the NumPy Structured array used for storing counters in `QTMTimer`
"""
MAX_ENTRIES: int = 500
"""Maximum number of timer/counter entries"""

# Check if code is running in parallel
COMM_WORLD, MPI_WORLD_SIZE, MPI_WORLD_RANK = None, 1, 0
if find_spec('mpi4py') is not None:
    from mpi4py.MPI import COMM_WORLD
    MPI_WORLD_SIZE = COMM_WORLD.Get_size()
    MPI_WORLD_RANK = COMM_WORLD.Get_rank()


class QTMTimer:
    """Timer Module of QuantumMASALA. Part of `QTMLogger`

    Manages a list of timers for tracking the wall-time of different sections
    of code. Provides a simple decorator to easily track the number of calls
    and total wall-time of the target function.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance with all handlers configured.

    Raises
    ------
    TypeError
        Raised if `logger` is not a `logging.Logger` instance

    Notes
    -----
    If running in interactive mode i.e, IPython/Jupyter Notebook, all timers
    are automatically disabled.
    """

    def __init__(self, logger: logging.Logger):
        if not isinstance(logger, logging.Logger):
            raise TypeError("'logger' must be a 'logging.Logger' instance. "
                            f"got {logger} (type {type(logger)})")
        self.logger: logging.Logger = logging.getLogger(LOGGER_NAME)
        """logger instance used to log events"""
        self.l_timer: np.ndarray = np.zeros(MAX_ENTRIES, dtype=TIMER_TYPE)
        """structured array containing timer data"""
        self.numtimer: int = 0
        """number of uniquely labeled timers"""

        # If running in interactive mode, disable timers by default.
        import sys
        self.enable_timer: bool = not hasattr(sys, 'ps1')
        """If False, the module is disabled; all methods have no effect"""

    def _find_timer(self, label: str) -> Optional[int]:
        """Returns index of timer with input label if present, else None.

        Parameters
        ----------
        label : str
            Label of the timer to search

        Returns
        -------
        idx_timer : Optional[int]
            Index of timer in `l_timer` if found else None.

        Raises
        ------
        TypeError
            Raised if `label` is not a string
        """
        if not isinstance(label, str):
            raise TypeError("'label' must be a string. "
                            f"got {label} (type {type(label)})")
        idx = np.nonzero(self.l_timer['label'][:self.numtimer] == label)[0]
        if len(idx) == 0:
            return None
        elif len(idx) == 1:
            return idx[0]
        else:
            raise RuntimeError("found multiple timers with same label. This is a bug")

    def _start_timer(self, label: str) -> None:
        """Starts a timer with input label, automatically initializing a new one
        if timer with given label does not exist.

        Parameters
        ----------
        label : str
            Label of the timer to start

        Raises
        -------
        ValueError
            Raised if the specified timer is already running.
        """
        itimer = self._find_timer(label)
        if itimer is None:
            self.l_timer[self.numtimer] = (label, 0, 0., False, 0.)
            itimer = self.numtimer
            self.numtimer += 1
            self.logger.info("timer '%s' created." % (label, ))

        timer = self.l_timer[itimer]
        if timer['status']:
            raise ValueError(f"timer '{label}' is already running.")
        timer['status'] = True
        timer['call'] += 1
        timer['start_time'] = perf_counter()
        self.logger.info("timer '%s' started." % (label, ))

    def _stop_timer(self, label: str) -> None:
        """Stops the timer with input label that is already running.

        Parameters
        ----------
        label : str
            Label of the timer to stop

        Raises
        -------
        ValueError
            Raised if the specified timer is not running.
        """
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")

        timer = self.l_timer[iclock]
        if not timer['status']:
            raise ValueError(f"timer '{label}' is not running.")
        timer['status'] = False
        delta = perf_counter() - timer['start_time']
        timer['time'] += delta
        self.logger.info("timer '%s' stopped. delta: %8.3f sec." % (label, delta))
        timer['start_time'] = 0

    def time(self, label: str):
        """Decorator for timing the runtimes of functions. Wraps the function with
        `_start_timer` and `_stop_timer` calls. Takes a string argument as label
        for timer

        This decorator is recommended over explicitly calling the timer start and stop
        methods to prevent mismatched calls (starting an already running timer /
        stopping a non-existent timer, etc.) and handle exceptions i.e, timer is stopped
        and the catched exception is rethrown.

        Parameters
        ----------
        label : str
            Label of the timer

        Notes
        -----
        Ideally, it would be nice for the decorator to not show up in trace stack
        during exceptions. But, I can't implement it. For reference:
        https://stackoverflow.com/questions/72146438/remove-decorator-from-stack-trace
        """
        import functools
        from sys import exc_info
        from _testcapi import set_exc_info

        def timer(func):
            @functools.wraps(func)
            def call_func(*args, **kwargs):
                if self.enable_timer:
                    self._start_timer(label)
                try:
                    out = func(*args, **kwargs)
                except:  # noqa : E722
                    tp, exc, tb = exc_info()
                    set_exc_info(tp, exc, tb)
                    del tp, exc, tb
                    raise
                finally:
                    if self.enable_timer:
                        self._stop_timer(label)
                return out
            return call_func
        return timer

    def reset_timer(self, label: str) -> None:
        """Resets the timer with input label.

        Parameters
        ----------
        label : str
            Label of the timer to stop

        Raises
        -------
        ValueError
            Raised if the specified timer does not exist.
        """
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")

        timer = self.l_timer[iclock]
        timer['status'] = False
        timer['call'], timer['time'] = 0, 0
        self.logger.info("timer '%s' cleared." % (label, ))

    def delete_timer(self, label: str) -> None:
        """Deletes the timer with input label.

        Parameters
        ----------
        label : str
            Label of the timer to stop

        Raises
        -------
        ValueError
            Raised if the specified timer does not exist.
        """
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")
        self.l_timer[iclock] = self.l_timer[self.numtimer - 1]
        self.numtimer -= 1
        self.logger.info("timer '%s' deleted." % (label, ))

    def __str__(self) -> str:
        """lists the status of all timers in a human-readable form"""
        out = f"{'TIMERS':^59}\n" + "-" * 59 + '\n' \
              f"|{'LABEL':^30}|{'CALL':^8}|{'TIME':^8}|{'STATUS':^9}|\n"
        out += "-" * 59 + '\n'
        if self.numtimer > 0:
            for count in self.l_timer[:self.numtimer]:
                out += (f"|{count['label']:^30}|{count['call']:8d}|"
                        f"{count['time']:8.2f}|"
                        f"{'RUNNING' if count['status'] else 'STOPPED':^9}|\n")
        out += "-" * 59 + '\n'
        return out


class QTMCounter:
    """Counter Module of QuantumMASALA. Part of `QTMLogger`

    Manages a list of counters for bookkeeping. Currently, it's usage is
    removed from the codebase.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance with all handlers configured.

    Raises
    ------
    TypeError
        Raised if `logger` is not a `logging.Logger` instance

    """

    def __init__(self, logger):
        if not isinstance(logger, logging.Logger):
            raise TypeError("'logger' must be a 'logging.Logger' instance. "
                            f"got {logger} (type {type(logger)})")
        self.logger: logging.Logger = logging.getLogger(LOGGER_NAME)
        """logger instance used to log events"""
        self.l_counter: np.ndarray = np.zeros(MAX_ENTRIES, dtype=COUNTER_TYPE)
        """structured array containing counter data"""
        self.numcounter: int = 0
        """number of uniquely labeled counters"""

    def _find_counter(self, label: str) -> Optional[bool]:
        idx = np.nonzero(self.l_counter['label'][:self.numcounter] == label)[0]
        if len(idx) == 0:
            return None
        elif len(idx) == 1:
            return idx[0]
        else:
            raise Exception("found multiple counters with same label. This is a bug")

    def add_to_counter(self, label: str, val: int = 1) -> None:
        icount = self._find_counter(label)
        if icount is None:
            self.l_counter[self.numcounter] = (label, 0)
            icount = self.numcounter
            self.numcounter += 1
            self.logger.info("counter '%s' created." % (label, ))

        counter = self.l_counter[icount]
        counter['count'] += val
        self.logger.info("counter '%s' updated: %7d -> %7d." %
                         (label, counter['count'] - val, counter['count']))

    def clear_counter(self, label: str) -> None:
        icount = self._find_counter(label)
        if icount is None:
            raise ValueError(f"counter '{label}' does not exist")

        counter = self.l_counter[icount]
        counter['count'] = 0
        self.logger.info("counter '%s' cleared." % (label, ))

    def delete_counter(self, label: str) -> None:
        icount = self._find_counter(label)
        if icount is None:
            raise ValueError(f"counter '{label}' does not exist")

        self.l_counter[icount] = self.l_counter[self.numcounter - 1]
        self.numcounter -= 1
        self.logger.info("counter '%s' deleted." % (label, ))

    def __str__(self):
        """lists the status of all counters in a human-readable form"""
        out = f"{'COUNTERS':^41}\n" + "-" * 41 + '\n' \
              f"|{'LABEL':^30}|{'COUNT':^8}|\n"
        out += "-" * 41 + '\n'
        if self.numcounter > 0:
            for count in self.l_counter[:self.numcounter]:
                out += f"|{count['label']:^30}|{count['count']:7d}|\n"
        out += "-" * 41 + '\n'
        return out


class QTMLogger(QTMTimer, QTMCounter):
    """QuantumMASALA's Logging Module

    When QuantumMASALA is imported, an instance of this class is generated
    and provides a global logging object that outputs to a .
    Provides methods to log messages. Inherits from
    `QTMTimer` and `QTMCounter`.

    To display the status of all timers and counters, pass the instance to the
    `print` function.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance with all handlers configured. By default, it is set to
        ``logging.getLogger(LOGGER_NAME)``

    """
    def __init__(self, logger=logging.getLogger(LOGGER_NAME)):
        QTMCounter.__init__(self, logger)
        QTMTimer.__init__(self, logger)
        self.setLevel(logging.INFO)

    def setLevel(self, level: int):  # noqa : N802
        """Alias of ``self.logger.setLevel(level)``"""
        self.logger.setLevel(level)

    def log(self, level: int, msg: str) -> None:
        """Alias of ``self.logger.log(level, msg)``
        """
        self.logger.log(level, msg)

    def debug(self, msg: str) -> None:
        """Alias of ``self.logger.info(msg)``
        """
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Alias of ``self.logger.debug(msg)``
        """
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Alias of ``self.logger.warning(msg)``
        """
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Alias of ``self.logger.error(msg)``
        """
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Alias of ``self.logger.critical(msg)``
        """
        self.logger.critical(msg)

    def exception(self, msg: str) -> None:
        """Alias of ``self.logger.exception(msg)``
        """
        self.logger.exception(msg)

    def __str__(self):
        return QTMTimer.__str__(self) + '\n' \
               + QTMCounter.__str__(self)


# Setting file handle to save log into
def qtmlogger_set_filehandle(file_dir: str, logging_level: int = logging.INFO,
                             capture_warnings: bool = True):
    """Sets up logging to file.

    The Logger used across QuantumMASALA is given by the module-level variable
    `LOGGER_NAME`. A `logging.FileHandler` instance is generated
    and configured to ouput log messages of level `logging_level` and above
    to file `file_dir`. If `capture_warnings` is True (default),
    `warnings.warn` messages are automatically logged to file.

    Parameters
    ----------
    file_dir : str
        Path of the log file
    logging_level : int, default=logging.INFO
        Logging threshold set to.
    capture_warnings : bool, default=True
        If True, warnings via `warnings.warn` function will also be logged
        to file.

    Notes
    -----
    If a file `file_dir` already exists, it will be renamed and a new file
    is created. The old file gets a '.preYYMMDD-HHMMSS' inserted in front
    of its suffix.

    `logging_level` does not affect the level set to the logger itself. By
    default, it is set to `logger.INFO`. So, to capture `logging.DEBUG`
    messages, `logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG)`
    must also be called.
    """
    logger = logging.getLogger(LOGGER_NAME)
    # Check if logger has existing handlers and warn if so
    if logger.hasHandlers():
        warnings.warn(f"logger '{LOGGER_NAME}' has handlers already initialized. "
                      "Clearing all existing handlers.")
        logger.handlers.clear()

    # Check if logfile already exists
    # If exists, rename it from filename.log -> filename.preYYMMDD-HHMMSS.log
    logfile_path = Path(file_dir)
    if logfile_path.is_file():
        if COMM_WORLD is not None:
            COMM_WORLD.barrier()
        if MPI_WORLD_RANK == 0:
            new_dir = logfile_path.with_suffix(
                    f'.pre{strftime("%Y%m%d-%H%M%S")}' + logfile_path.suffix
                )
            warnings.warn(f"log file '{file_dir}' already exists. "
                          f"Renaming it to '{new_dir}'")
            logfile_path.rename(new_dir)

    logger_file = logging.FileHandler(file_dir)
    logger_file.setLevel(logging_level)

    logger_fmt = logging.Formatter(
        LOG_FORMAT.format(MPI_WORLD_RANK=MPI_WORLD_RANK, MPI_WORLD_SIZE=MPI_WORLD_SIZE)
    )
    logger_file.setFormatter(logger_fmt)
    logger.addHandler(logger_file)

    logging.captureWarnings(capture_warnings)
    if capture_warnings:
        warn_logger = logging.getLogger('py.warnings')
        warn_logger.addHandler(logger_file)


qtmlogger = QTMLogger()
"""Global Instance of `QTMLogger`"""

# Disabling logging here. It will be enabled by qtmconfig when importing the package.
logging.disable()

warn = qtmlogger.warning
"""Alias of ``qtmlogger.warning``"""
