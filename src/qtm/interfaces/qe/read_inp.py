from typing import NamedTuple, Optional, Union, Any


class Control(NamedTuple):
    calculation: str = "scf"
    title: str = ""
    verbosity: str = "low"
    restart_mode: str = "from_scratch"
    iprint: int = 100000
    outdir: str = "./"
    prefix: str = "QuantumMASALA"
    max_seconds: float = 1e7
    disk_io: str = "low"
    pseudo_dir: str = "./"


class System(NamedTuple):
    ibrav: int

    nat: int
    ntyp: int
    ecutwfc: float

    celldm: dict[int, float] = {}
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    cosab: Optional[float] = None
    cosac: Optional[float] = None
    cosbc: Optional[float] = None

    ecutrho: Optional[float] = None
    nr1: Optional[int] = None
    nr2: Optional[int] = None
    nr3: Optional[int] = None

    nspin: int = 1
    noncolin: bool = False
    nbnd: Optional[int] = None
    starting_magnetization: dict[int, float] = {}
    occupations: str = "fixed"
    degauss: float = 0.0
    smearing: str = "gaussian"

    nosym: bool = False
    nosym_evc: bool = False
    noinv: bool = False
    input_dft: Optional[str] = None


class Electrons(NamedTuple):
    electron_maxstep: int = 100
    conv_thr: float = 1e-6
    mixing_mode: str = "plain"
    mixing_beta: float = 0.7
    mixing_ndim: int = 8
    diagonalization: str = "david"
    diago_thr_init: float = 1e-2
    diago_david_ndim: int = 4
    diago_full_acc: bool = False
    startingpot: str = "atomic"
    startingwfc: str = "random"


class Ions(NamedTuple):
    pass


class Cell(NamedTuple):
    pass


class FCP(NamedTuple):
    pass


class RISM(NamedTuple):
    pass


class PWscfIn(NamedTuple):
    control: Control
    system: System
    electrons: Electrons

    atomic_species: tuple[None, list[tuple[str, float, str]]]
    atomic_positions: tuple[str, list[tuple[str, float, float, float]]]
    k_points: tuple[str, list[str]]
    cell_parameters: tuple[str, list[tuple[float, float, float]]] = None

    ions: Ions = Ions()
    cell: Cell = Cell()
    fcp: FCP = FCP()
    rism: RISM = RISM()

    @classmethod
    def from_file(cls, fname: str):
        data_all = {}
        # Reading the file
        try:
            with open(fname, "r", encoding="utf8") as file:
                text = file.readlines()
        except FileNotFoundError:
            raise ValueError(f"PWscf Input file '{fname}' not found")

        # Function to iterate over lines read from file
        numlines = len(text)
        iline = 0

        def read_line(case_sensitive_=False):
            nonlocal iline
            if iline >= numlines:
                return ""

            if not case_sensitive_:
                line_ = text[iline].lower().strip()
            else:
                line_ = text[iline].strip()
            iline += 1
            return line_

        while iline < numlines:
            line = read_line()
            # Skipping Empty lines
            if line.strip() == "":
                continue

            # First Namelists are parsed
            if line.startswith("&"):
                namelist = line[1:]
                if namelist not in cls.__annotations__:
                    raise ValueError(f"Nammelist '{namelist}' not recognised")

                # Getting all supported arguments of namelist
                namelist_typ = cls.__annotations__[namelist]
                data_typ = cls.__annotations__[namelist].__annotations__

                # Parsing arguments in namelist
                data = {}
                while True:
                    line = read_line().rstrip(",")
                    # Discard Comments
                    if line.lstrip()[0] in ["!", "#"]:
                        continue
                    # Namelist ends when line starts with '/'
                    elif line.lstrip()[0] == "/":
                        break

                    # Evaluating each 'key=value' substring that are seperated by commas
                    for parval in line.split(","):
                        # Splitting it into two
                        parval = parval.split("=")
                        if len(parval) != 2:
                            raise ValueError(
                                f'Cannot parse "{parval}" to (key, value) pair'
                            )
                        par, val = parval

                        par, val = par.strip(), val.strip().strip("'")
                        # If parameter is part of a list like 'param(i)', then get the index i
                        idx = None
                        if "(" in par:
                            par, idx = par.split("(")
                            idx = idx.rstrip(")")

                        # Function to parse string to given type; Supports Fortran Repr
                        def parse_typ(typ_: Union[type, Optional[Any]], val_: str):
                            # If argument is 'Optional' get its type
                            if typ_.__name__ == "Optional":
                                typ_ = typ_.__args__[0]
                            # Fortran Booleans
                            if typ_ is bool:
                                if val_ == ".true":
                                    val_ = True
                                elif val_ == ".false":
                                    val_ = False
                                else:
                                    raise ValueError(f"Cannot parse '{val_}' to bool")
                            # Fortran Real
                            elif typ_ is float:
                                val_ = val_.replace("d", "e")

                            try:
                                return typ_(val_)
                            except ValueError:
                                raise ValueError(
                                    f"cannot parse '{val_}' to {typ_.__name__}"
                                )

                        if par in data_typ:
                            typ = data_typ[par]
                            if typ.__name__ != "dict":
                                data[par] = parse_typ(typ, val)
                            else:
                                if par not in data:
                                    data[par] = {}
                                typ_key, typ_val = typ.__args__
                                idx = parse_typ(typ_key, idx)
                                data[par][idx] = parse_typ(typ_val, val)
                        else:
                            raise ValueError(
                                f"Parameter '{par}' not valid or supported"
                            )

                try:
                    data_all[namelist] = namelist_typ(**data)
                except ValueError:
                    raise ValueError(
                        f"Failed to generate Namelist Object for '{namelist}'"
                    )

            # Now, Cards are processed
            else:
                cardopt = line.split("{" if "{" in line else None)
                if len(cardopt) == 0 or len(cardopt) > 2:
                    raise ValueError(f'Cannot parse Card Header "{cardopt}"')
                card = cardopt[0].strip().lower()
                option = (
                    cardopt[1].rstrip("}").strip().lower()
                    if len(cardopt) == 2
                    else None
                )
                option_typ, data_typ = cls.__annotations__[card].__args__
                data_typ = data_typ.__args__[0]
                if option_typ is not None:
                    option = option_typ(option)

                data = []

                case_sensitive = False
                if card == "atomic_species":
                    n_lines_card = data_all["system"].ntyp
                    if n_lines_card <= 0:
                        raise ValueError(
                            f"value of 'ntyp' must be a positive integer. Got {n_lines_card}"
                        )
                    case_sensitive = True
                elif card == "atomic_positions":
                    n_lines_card = data_all["system"].nat
                    if n_lines_card <= 0:
                        raise ValueError(
                            f"value of  'nat' must be a positive integer. Got {n_lines_card}"
                        )
                    case_sensitive = True
                elif card == "k_points":
                    if option == "automatic":
                        n_lines_card = 1
                    else:
                        n_lines_card = int(read_line())
                        if n_lines_card <= 0:
                            raise ValueError(
                                f"number of k-points must be positive integer. Got {n_lines_card}"
                            )
                elif card == "cell_parameters":
                    n_lines_card = 3
                else:
                    raise ValueError(f"card '{card}' not recognised")

                for il in range(n_lines_card):
                    line = read_line(case_sensitive)
                    if line == "":
                        raise ValueError(
                            f"card '{card.upper()}' contains less lines than expected. "
                            f"Expected {n_lines_card} lines. Got {il}"
                        )
                    if data_typ is not str:
                        line = line.split()
                        for i, typ in enumerate(data_typ.__args__):
                            line[i] = typ(line[i])
                    data.append(line)
                data_all[card] = (option, data)

        return cls(**data_all)
