"""Factory for constructing (possibly constrained) 1D Galerkin function spaces.

High-level helper that:
  * Builds a pure orthogonal / polynomial space when no boundary conditions
    (bcs) are supplied.
  * Applies (homogeneous) boundary conditions by creating a Composite space
    whose basis satisfies the constraints.
  * For inhomogeneous boundary data returns a DirectSum of:
        Composite (homogeneous constrained space)
        +
        BCGeneric (boundary lifting space providing fixed values)

This keeps the user-facing API minimal: specify N, the base space class
(e.g. Chebyshev, Legendre, Jacobi, Fourier, Composite), and an optional
boundary condition dictionary.

Boundary condition dictionary format (keys 'left' / 'right'):
    {
      "left":  {"D": 0, "N": 1},
      "right": {"D": 0}
    }
Keys like "D", "N", "N2", "R", "W" correspond to Dirichlet, first/second
derivatives, Robin, weighted, etc., as interpreted by BoundaryConditions.
"""

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
    """Return a (possibly boundary-constrained) function space instance.

    If bcs is None:
        Returns an unconstrained instance of the provided base `space`
        class (e.g. Chebyshev(N), Legendre(N), etc.).

    If bcs is provided (dict or BoundaryConditions):
        1. Wrap into BoundaryConditions.
        2. Build homogeneous Composite space C (basis satisfies BCs with
           zero boundary values).
        3. If BCs are homogeneous, return C.
        4. Else build BCGeneric lifting space B (encodes inhomogeneous
           boundary values) and return DirectSum(C, B).

    Args:
        N: Number of modes (polynomials) or base functions (unconstrained).
        space: Base orthogonal / spectral space class (callable) or already
            constructed BaseSpace subclass (factory expected signature:
            space(N, domain=..., system=..., name=..., fun_str=..., **kw)).
        bcs: BoundaryConditions instance or raw dict specifying left/right
            boundary constraints. See module docstring for format.
        domain: Physical Domain (maps to reference domain internally).
        system: Optional CoordSys for curvilinear mappings.
        name: Name of the resulting space (used for symbolic function ids).
        fun_str: Basis function symbol stem.
        **kw: Extra keyword arguments forwarded to the base `space`
            constructor (e.g. alpha/beta for Jacobi, scaling, stencil).

    Returns:
        BaseSpace | Composite | DirectSum:
            - BaseSpace (unconstrained) if bcs is None
            - Composite (homogeneous constrained) if BCs homogeneous
            - DirectSum(Composite, BCGeneric) if BCs inhomogeneous

    Examples:
        Homogeneous Dirichlet:
            V = FunctionSpace(32, Chebyshev, bcs={'left': {'D': 0},
                                                  'right': {'D': 0}})
        Inhomogeneous Dirichlet:
            V = FunctionSpace(32, Chebyshev, bcs={'left': {'D': 1},
                                                  'right': {'D': 0}})

        No boundary conditions:
            V = FunctionSpace(32, Chebyshev)
    """
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
            bcs.num_bcs(),
            space,
            bcs=bcs,
            domain=domain,
            system=system,
            num_quad_points=N,
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
