__all__ = ["PseudoPotFile"]

import os
from hashlib import md5
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PseudoPotFile(ABC):
    """Abstract Base class as template for Pseudopotential Reader.

    Pseudopotential readers will inherit this class and implement
    `read(dirname)` method for reading pseudopotential files.

    Attributes
    ----------
    label : str
        User-provided label for atom type described by the pseudopotential.
    dirname : str
        Path of the data file.
    filename : str
        Name of the file `os.path.basename`.
    md5_checksum :
        MD5 Hash of the data file.
    """
    dirname: str
    filename: str = field(init=False)
    md5_checksum: str = field(init=False)

    valence: int
    libxc_func: tuple[str, str]

    @classmethod
    @abstractmethod
    def from_file(cls, dirname: str, valence: int):
        return

    def __post_init__(self):
        self.filename = os.path.basename(self.dirname)

        hash_md5 = md5()
        with open(self.dirname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        self.md5_checksum = hash_md5.hexdigest()
