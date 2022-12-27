from time import perf_counter
import numpy as np


counter_type = np.dtype([('label', 'U30'), ('call', 'i8'), ('time', 'f8'),
                         ('status', 'b'), ('start_time', 'f8')])
MAX_ENTRIES = 100


class PWCounter:

    def __init__(self):
        self.l_count = np.zeros(MAX_ENTRIES, dtype=counter_type)
        self.numcount = 0

    def start_clock(self, label):
        idxcount = np.nonzero(self.l_count['label'][:self.numcount] == label)[0]
        if len(idxcount) == 0:
            self.l_count[self.numcount] = (label, 0, 0., False, 0.)
            counter = self.l_count[self.numcount]
            self.numcount += 1
        elif len(idxcount) != 1:
            raise ValueError("found multiple counters with same label. "
                             "This is a bug.")
        else:
            counter = self.l_count[idxcount[0]]

        if counter['status']:
            raise ValueError(f"counter '{label}' is already running.")
        counter['status'] = True
        counter['call'] += 1
        counter['start_time'] = perf_counter()

    def stop_clock(self, label):
        idxcount = np.nonzero(self.l_count['label'][:self.numcount] == label)[0]
        if len(idxcount) == 0:
            raise ValueError(f"counter '{label}' does not exist.")
        idxcount = idxcount[0]

        counter = self.l_count[idxcount]
        if not counter['status']:
            raise ValueError(f"counter '{label}' is not running.")
        counter['status'] = False
        counter['time'] += perf_counter() - counter['start_time']
        counter['start_time'] = 0

    def __str__(self):
        out = f"|{'LABEL':^30}|{'CALL':^7}|{'TIME':^8}|{'STATUS':^9}|\n"
        out += "-" * 49 + '\n'
        for count in self.l_count[:self.numcount]:
            out += (f"|{count['label']:^30}|{count['call']:7d}|"
                    f"{count['time']:8.2f}|"
                    f"{'RUNNING' if count['status'] else 'STOPPED':^9}|\n")
        return out


pw_counter = PWCounter()
