
import logging
from time import perf_counter, strftime
from typing import Optional
import numpy as np

from importlib.util import find_spec

MPI_WORLD_SIZE, MPI_WORLD_RANK = 1, 0
if find_spec('mpi4py') is not None:
    from mpi4py.MPI import COMM_WORLD
    MPI_WORLD_SIZE = COMM_WORLD.Get_size()
    MPI_WORLD_RANK = COMM_WORLD.Get_rank()


timer_type = np.dtype([('label', 'U30'), ('call', 'i8'), ('time', 'f8'),
                       ('status', 'b'), ('start_time', 'f8')])
counter_type = np.dtype([('label', 'U30'), ('count', 'i8')])
MAX_ENTRIES = 100

logger = logging.getLogger('QTMPy')
logger_file = logging.FileHandler(f'QTMPy_{strftime("%Y%m%d-%H%M%S")}.log')
logger_file.setLevel(logging.INFO)
logger_fmt = logging.Formatter(f'%(asctime)s - '
                               f'proc {MPI_WORLD_RANK}/{MPI_WORLD_SIZE} - '
                               f'%(message)s')

logger_file.setFormatter(logger_fmt)
# logger.addHandler(logger_file)
logger.setLevel(logging.INFO)


class PWTimer:

    def __init__(self):
        self.l_timer = np.zeros(MAX_ENTRIES, dtype=timer_type)
        self.numtimer = 0

    def _find_timer(self, label: str) -> Optional[bool]:
        idx = np.nonzero(self.l_timer['label'][:self.numtimer] == label)[0]
        if len(idx) == 0:
            return None
        elif len(idx) == 1:
            return idx[0]
        else:
            raise Exception("found multiple timers with same label. This is a bug")

    def start_timer(self, label: str) -> None:
        itimer = self._find_timer(label)
        if itimer is None:
            self.l_timer[self.numtimer] = (label, 0, 0., False, 0.)
            itimer = self.numtimer
            self.numtimer += 1
            logger.debug("timer '%s' created." % (label, ))

        timer = self.l_timer[itimer]
        if timer['status']:
            raise ValueError(f"timer '{label}' is already running.")
        timer['status'] = True
        timer['call'] += 1
        timer['start_time'] = perf_counter()
        logger.debug("timer '%s' started." % (label, ))

    def stop_timer(self, label: str) -> None:
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")

        timer = self.l_timer[iclock]
        if not timer['status']:
            raise ValueError(f"timer '{label}' is not running.")
        timer['status'] = False
        delta = perf_counter() - timer['start_time']
        timer['time'] += delta
        logger.debug("timer '%s' stopped. delta: %8.3f sec." % (label, delta))
        timer['start_time'] = 0

    def clear_timer(self, label: str) -> None:
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")

        timer = self.l_timer[iclock]
        timer['status'] = False
        timer['call'], timer['time'] = 0, 0
        logger.debug("timer '%s' cleared." % (label, ))

    def delete_timer(self, label: str) -> None:
        iclock = self._find_timer(label)
        if iclock is None:
            raise ValueError(f"timer '{label}' does not exist.")
        self.l_timer[iclock] = self.l_timer[self.numtimer - 1]
        self.numtimer -= 1
        logger.debug("timer '%s' deleted." % (label, ))

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
            logger.debug("counter '%s' created." % (label, ))

        counter = self.l_counter[icount]
        counter['count'] += val
        logger.debug("counter '%s' updated: %7d -> %7d." %
                    (label, counter['count'] - val, counter['count']))

    def clear_counter(self, label: str) -> None:
        icount = self._find_counter(label)
        if icount is None:
            raise ValueError(f"counter '{label}' does not exist")

        counter = self.l_counter[icount]
        counter['count'] = 0
        logger.debug("counter '%s' cleared." % (label, ))

    def delete_counter(self, label: str) -> None:
        icount = self._find_counter(label)
        if icount is None:
            raise ValueError(f"counter '{label}' does not exist")

        self.l_counter[icount] = self.l_counter[self.numcounter - 1]
        self.numcounter -= 1
        logger.debug("counter '%s' deleted." % (label, ))

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
    __slots__ = ['l_timer', 'numtimer', 'l_counter', 'numcounter']

    def __init__(self):
        PWCounter.__init__(self)
        PWTimer.__init__(self)

    def __str__(self):
        return PWTimer.__str__(self) + '\n' \
               + PWCounter.__str__(self)


pw_counter = PWLogger()
