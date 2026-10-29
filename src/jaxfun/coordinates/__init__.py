"""
Curvilinear and Cartesian coordinate systems.

`jaxfun.coordinates.system` holds the coordinate systems (`CoordSys`) and
`jaxfun.coordinates.symbols` the SymPy symbols they own: base scalars, time,
base vectors and dyadics, and metric symbols.
"""

from jaxfun.coordinates.symbols import (
    BaseDyadic as BaseDyadic,
    BaseScalar as BaseScalar,
    BaseTime as BaseTime,
    BaseVector as BaseVector,
    ContravariantBaseVector as ContravariantBaseVector,
    MetricComponent as MetricComponent,
    MetricDeterminant as MetricDeterminant,
    MetricSymbol as MetricSymbol,
    _is_dyadiclike as _is_dyadiclike,
    _is_tensorlike as _is_tensorlike,
    _is_vectorlike as _is_vectorlike,
    latex_symbol as latex_symbol,
)
from jaxfun.coordinates.system import (
    CartCoordSys as CartCoordSys,
    CoordSys as CoordSys,
    R as R,
    SubCoordSys as SubCoordSys,
    get_CoordSys as get_CoordSys,
    get_system as get_system,
    t as t,
    x as x,
    y as y,
    z as z,
)
