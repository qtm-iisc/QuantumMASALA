"""Simple, DFT-free unit tests for `qtm.mpi.comm.QTMComm`'s
control-flow/context-manager semantics: these are rank-count-independent
(they don't depend on actually having more than one process), so they're
fully testable in this single-process environment, unlike the cross-process
*data* consistency checks in `qtm.mpi.check_args` or the real distributed
splitting/gathering in `qtm.mpi.gspace`/`qtm.mpi.containers` (those compare
each rank's data against rank 0's broadcast value, which is trivially
"equal to itself" with only one rank -- meaningless to unit test here).

Uses `QTMComm.comm_self` (`mpi4py.MPI.COMM_SELF`, a real size-1
communicator) and `QTMComm.comm_null` (`mpi4py.MPI.COMM_NULL`) directly, the
same pattern the class uses internally.
"""
import pytest

from qtm.mpi.comm import QTMComm, split_comm_pwgrp


def test_normal_context_manager_exit_does_not_raise():
    comm = QTMComm(QTMComm.comm_self)
    with comm:
        pass


def test_exception_inside_context_is_wrapped_with_original_chained():
    # QTMComm.__exit__ re-raises any exception from inside the 'with' block
    # as a generic Exception (to keep multi-process barriers consistent),
    # chaining the original as '__cause__' -- already relied on in
    # '../dft_tests/test_occup.py::test_fixed_occ_rejects_odd_numel'.
    comm = QTMComm(QTMComm.comm_self)
    with pytest.raises(Exception) as exc_info:
        with comm:
            raise ValueError("boom")
    assert type(exc_info.value) is Exception
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_skip_with_block_suppresses_exception_and_skips_rest_of_block():
    null_comm = QTMComm(QTMComm.comm_null)
    assert null_comm.is_null

    reached_after_skip = False
    with null_comm as comm:
        comm.skip_with_block()
        reached_after_skip = True  # must never execute
    assert not reached_after_skip  # 'with' block exited cleanly, no exception escaped


def test_null_comm_disables_all_methods_except_skip_with_block():
    null_comm = QTMComm(QTMComm.comm_null)
    with pytest.raises(AttributeError):
        null_comm.Bcast
    with pytest.raises(AttributeError):
        null_comm.bcast
    null_comm.skip_with_block  # must NOT raise -- explicitly exempted

    # Plain (non-method) attributes remain accessible even when null.
    assert null_comm.is_null is True
    assert null_comm.size == 0


def test_root_gives_a_rank_zero_communicator():
    comm = QTMComm(QTMComm.comm_self)  # size 1, rank 0
    root = comm.Root()
    assert not root.is_null
    assert root.size == 1 and root.rank == 0


def test_incl_on_single_process_comm_short_circuits_to_self():
    comm = QTMComm(QTMComm.comm_self)
    assert comm.Incl([0]) is comm
    with pytest.raises(ValueError):
        comm.Incl([1])  # only rank 0 exists in a size-1 communicator


def test_split_comm_pwgrp_with_pwgrp_size_one_on_single_process():
    comm = QTMComm(QTMComm.comm_self)
    pwgrp_comm, intercomm = split_comm_pwgrp(comm, pwgrp_size=1)
    assert pwgrp_comm is None  # no intra-pwgrp communicator needed when size 1
    assert not intercomm.is_null and intercomm.size == 1


def test_split_comm_pwgrp_rejects_non_divisor_pwgrp_size():
    comm = QTMComm(QTMComm.comm_self)  # size 1
    with pytest.raises(ValueError):
        split_comm_pwgrp(comm, pwgrp_size=2)  # 2 does not evenly divide 1
