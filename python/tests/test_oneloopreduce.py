"""Interpreter-level tests for `symbolica.community.oneloopreduce`.

The only coverage of the FFI boundary; the Rust suite cannot reach it. Needs the
module built into a symbolica-community root (README, "Wiring it into
symbolica-community"), so it does not run in CI. Run it by hand, unparallelised
-- Symbolica aborts when an unlicensed instance is touched from a second thread:

    SYMBOLICA_HIDE_BANNER=1 pytest python/tests/test_oneloopreduce.py
"""

import pytest
from symbolica import E, Expression, S
from symbolica.community.oneloopreduce import (
    IntegralFamily, MasterIntegral, Propagator, Reduction, initialize_module,
)


def test_the_module_is_initialized():
    initialize_module()  # idempotent; the facade's __init__ already ran it
    assert all((Propagator, IntegralFamily, Reduction, MasterIntegral))


def test_dot_kept_its_symmetric_linear_attributes():
    # Symbolica fixes a symbol's attributes at first mention and rejects a
    # conflicting redeclaration, so this call succeeding *is* the assertion.
    dot = S("oneloopreduce::dot", is_symmetric=True, is_linear=True)
    k, q1, q2 = S("oneloopreduce::k"), S("oneloopreduce::q1"), S("oneloopreduce::q2")
    assert str(dot(q1, k)) == str(dot(k, q1))
    assert str(dot(k, q1 + q2)) == str(dot(k, q1) + dot(k, q2))
    with pytest.raises(TypeError):
        S("oneloopreduce::dot", is_symmetric=False)


def test_massless_bubble_reduces_to_one_master():
    bubble = IntegralFamily([Propagator(E("0"))] * 2, [E("s")])
    ((coefficient, master),) = bubble.reduce().simplify().terms
    assert (master.kind, master.head) == ("bubble", "B0")
    assert [str(a) for a in master.arguments] == ["s", "0", "0"]
    assert str(coefficient) == "1"


def test_an_expression_numerator_crosses_the_boundary():
    dot, k, q1 = S("oneloopreduce::dot"), S("oneloopreduce::k"), S("oneloopreduce::q1")
    kin = [E("p1sq"), E("s"), E("p2sq")]
    triangle = IntegralFamily([Propagator(E("msq"))] * 3, kin, numerator=dot(k, q1))
    reduction = triangle.reduce().simplify()
    assert sorted(m.kind for _, m in reduction.terms) == ["bubble", "bubble", "triangle"]
    assert sorted(str(c) for c, _ in reduction.terms) == ["-1/2", "-1/2*p1sq", "1/2"]
    assert isinstance(reduction.to_expression(), Expression)


def test_a_malformed_family_raises():
    with pytest.raises(ValueError):
        IntegralFamily([Propagator(E("0"))] * 2, [])  # a bubble has one invariant


def test_the_recursion_guard_raises_instead_of_aborting():
    # Overrunning the stack is a SIGABRT no `except` could catch. The bound
    # lives in the library, so the constructor accepts this and reduce() refuses.
    huge = IntegralFamily([Propagator(E("0"))] * 2, [E("s")], exponents=[40, 1])
    with pytest.raises(ValueError):
        huge.reduce()
