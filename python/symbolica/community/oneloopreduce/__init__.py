"""
# One-loop reduce

IBP reduction of one-loop Feynman integrals to the four scalar master
integrals `A0`, `B0`, `C0` and `D0`.

Kinematics, numerators and coefficients all cross the boundary as Symbolica
`Expression`s, so a reduction composes with the rest of Symbolica without a
round trip through strings.

# Symbols

The module works in the `oneloopreduce::` namespace. The numerator is a
polynomial in the symmetric, linear dot product `oneloopreduce::dot` over the
loop momentum `oneloopreduce::k` and the external momenta
`oneloopreduce::q1`, `q2`, ....

# Example

```python
from symbolica import E, S
from symbolica.community.oneloopreduce import IntegralFamily, Propagator

# A massless bubble with an off-shell external leg.
bubble = IntegralFamily(
    propagators=[Propagator(E("0")), Propagator(E("0"))],
    invariants=[E("s")],
)

reduction = bubble.reduce()
for coefficient, master in reduction.terms:
    print(master.kind, master.arguments, "*", coefficient)

print(reduction.to_expression())

# A rank-one numerator on a massive triangle.
dot = S("oneloopreduce::dot")
k = S("oneloopreduce::k")
q1 = S("oneloopreduce::q1")

triangle = IntegralFamily(
    propagators=[Propagator(E("msq"))] * 3,
    invariants=[E("p1sq"), E("s"), E("p2sq")],
    numerator=dot(k, q1),
)
print(triangle.reduce().simplify().to_expression())
```

# Contributors
- Elijah Cavan
"""

from ..oneloopreduce_native import *

initialize_module()
