__all__ = ['pw_logger', 'logger_set_filehandle']

import logging
from pathlib import Path
import warnings
from time import perf_counter, strftime
from importlib.util import find_spec
from typing import Optional

import numpy as np


MPI_WORLD_SIZE, MPI_WORLD_RANK = 1, 0
if find_spec('mpi4py') is not None:
    from mpi4py.MPI import COMM_WORLD
    MPI_WORLD_SIZE = COMM_WORLD.Get_size()
    MPI_WORLD_RANK = COMM_WORLD.Get_rank()


timer_type = np.dtype([('label', 'U30'), ('call', 'i8'), ('time', 'f8'),
                       ('status', 'b'), ('start_time', 'f8')])
counter_type = np.dtype([('label', 'U30'), ('count', 'i8')])
MAX_ENTRIES = 100

LOG_FMT = '%(asctime)s - proc {world_rank}/{world_size}: %(message)s'


def _get_logger():
    logger = logging.getLogger('QTMPy')
    logger.setLevel(logging.INFO)
    return logger


def logger_set_filehandle(file_name: str):
    logger = logging.getLogger('QTMPy')
    if logger.hasHandlers():
        pw_logger.warn("'pw_logger' has handlers already initialized.")

    if MPI_WORLD_RANK == 0:
        logfile_path = Path(file_name)
        if logfile_path.is_file():
            logfile_path.rename(
                logfile_path.with_suffix(
                    f'.pre{strftime("%Y%m%d-%H%M%S")}' + logfile_path.suffix
                ))

    logger_file = logging.FileHandler(file_name)
    logger_file.setLevel(logging.DEBUG)
    logger_fmt = logging.Formatter(
        LOG_FMT.format(world_rank=MPI_WORLD_RANK, world_size=MPI_WORLD_SIZE)
    )
    logger_file.setFormatter(logger_fmt)
    logger.addHandler(logger_file)


class PWTimer:

    def __init__(self):
        self.logger = _get_logger()
        self.l_timer = np.zeros(MAX_ENTRIES, dtype=timer_type)
        self.numtimer = 0
        import sys
        self.enabled = not hasattr(sys, 'ps1')

    def _find_timer(self, label: str) -> Optional[int]:
        if not isinstance(label, str):
            raise ValueError("'label' must be a string. "
                             f"got {label} (type {type(label)})")
        idx = np.nonzero(self.l_timer['label'][:self.numtimer] == label)[0]
        if len(idx) == 0:
            return None
        elif len(idx) == 1:
            return idx[0]
        else:
            raise Exception("found multiple timers with same label. This is a bug")

    def _start_timer(self, label: str) -> None:
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
        if self.enabled:
            def timer(func):
                def call_func(*args, **kwargs):
                    self._start_timer(label)
                    try:
                        out = func(*args, **kwargs)
                    except BaseException as e:
                        self._stop_timer(label)
                        raise e
                    self._stop_timer(label)
                    return out
                return call_func
        else:
            def timer(func):
                def call_func(*args, **kwargs):
                    return func(*args, **kwargs)
                return call_func
        return timer

    def clear_timer(self, label: str) -> None:
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")

        timer = self.l_timer[iclock]
        timer['status'] = False
        timer['call'], timer['time'] = 0, 0
        self.logger.info("timer '%s' cleared." % (label, ))

    def delete_timer(self, label: str) -> None:
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")
        self.l_timer[iclock] = self.l_timer[self.numtimer - 1]
        self.numtimer -= 1
        self.logger.info("timer '%s' deleted." % (label, ))

    def __str__(self):
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


class PWCounter:

    def __init__(self):
        self.logger = _get_logger()
        self.l_counter = np.zeros(MAX_ENTRIES, dtype=counter_type)
        self.numcounter = 0

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
        out = f"{'COUNTERS':^41}\n" + "-" * 41 + '\n' \
              f"|{'LABEL':^30}|{'COUNT':^8}|\n"
        out += "-" * 41 + '\n'
        if self.numcounter > 0:
            for count in self.l_counter[:self.numcounter]:
                out += f"|{count['label']:^30}|{count['count']:7d}|\n"
        out += "-" * 41 + '\n'
        return out


class PWLogger(PWTimer, PWCounter):
    __slots__ = ['pwcomm', 'l_timer', 'numtimer', 'l_counter', 'numcounter']

    def __init__(self):
        PWCounter.__init__(self)
        PWTimer.__init__(self)

    def warn(self, msg: str):
        if MPI_WORLD_RANK == 0:
            warnings.warn(msg)
        self.log_message(msg)

    def log_message(self, msg: str):
        self.logger.info(msg)

    def __str__(self):
        return PWTimer.__str__(self) + '\n' \
               + PWCounter.__str__(self)


pw_logger = PWLogger()
