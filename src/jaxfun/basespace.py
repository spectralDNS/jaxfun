import sympy as sp

from jaxfun.coordinates import CartCoordSys, CoordSys, x

n = sp.Symbol("n", integer=True, positive=True)  # index


class BaseSpace:
    """Abstract base class for function spaces.

    Provides a minimal interface shared by concrete function / spectral /
    neural network spaces. A coordinate system is attached so symbolic
    operators (gradient, divergence, etc.) can be expressed consistently.

    Args:
        system: Coordinate system associated with the space. If None, a
            1D Cartesian system is created automatically.
        name: Human-readable name of the space (used in repr / debugging).
        fun_str: Base string used when generating symbolic function names.

    Attributes:
        name (str): Name of the space.
        fun_str (str): Symbolic function name stem (e.g. 'u', 'phi').
        system (CoordSys): Underlying coordinate system (Cartesian by default).
        is_transient (bool): Whether the space includes time as a coordinate
            (default False here; subclasses may override).
    """

    is_transient = False

    def __init__(
        self,
        system: CoordSys | None = None,
        name: str = "BaseSpace",
        fun_str: str = "phi",
    ) -> None:
        self.name = name
        self.fun_str = fun_str
        self.system: CoordSys = CartCoordSys("N", (x,)) if system is None else system
