import sympy as sp
from jaxfun.composite import Composite, DirectSum, BCGeneric
from jaxfun.Basespace import BaseSpace, BoundaryConditions, Domain
from jaxfun.coordinates import CoordSys


def FunctionSpace(
    N: int,
    space: BaseSpace,
    bcs: BoundaryConditions | dict = None,
    domain: Domain = None,
    system: CoordSys = None,
    name: str = "fun",
    fun_str: str = "psi",
    **kw,
):
    
    if bcs is not None:
        bcs = BoundaryConditions(bcs, domain=domain)
        C = Composite(
            N,
            space,
            bcs=bcs.get_homogeneous(),
            domain=domain if domain is not None else (-1, 1),
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
            domain=domain if domain is not None else (-1, 1),
            system=system,
            M = N,
            name=name + "_b",
            fun_str=fun_str + "_b",
        )
        return DirectSum(C, B)
    return space(
        N,
        domain=domain if domain is not None else (-1, 1),
        system=system,
        name=name,
        fun_str=fun_str,
        **kw,
    )
