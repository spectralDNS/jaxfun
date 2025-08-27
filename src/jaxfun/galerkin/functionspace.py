from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain

from .composite import (
    BCGeneric,
    BoundaryConditions,
    Composite,
    DirectSum,
)


def FunctionSpace(
    N: int,
    space: BaseSpace,
    bcs: BoundaryConditions | dict = None,
    domain: Domain = None,
    system: CoordSys = None,
    name: str = "fun",
    fun_str: str = "psi",
    **kw,
) -> BaseSpace | DirectSum | Composite:
    if bcs is not None:
        bcs = BoundaryConditions(bcs, domain=domain)
        C = Composite(
            N,
            space,
            bcs=bcs.get_homogeneous(),
            domain=domain,
            name=name,
            fun_str=fun_str,
            system=system,
            **kw,
        )
        if bcs.is_homogeneous():
            return C
        B = BCGeneric(
            bcs.num_bcs() - 1,
            space,
            bcs=bcs,
            domain=domain,
            system=system,
            M=N,
            name=name + "_b",
            fun_str=fun_str + "_b",
        )
        return DirectSum(C, B)
    return space(
        N,
        domain=domain,
        system=system,
        name=name,
        fun_str=fun_str,
        **kw,
    )
