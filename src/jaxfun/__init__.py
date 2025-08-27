from . import galerkin as galerkin, pinns as pinns
from .basespace import BaseSpace as BaseSpace
from .coordinates import CoordSys as CoordSys, get_CoordSys as get_CoordSys
from .operators import (
    Cross as Cross,
    Curl as Curl,
    Div as Div,
    Dot as Dot,
    Grad as Grad,
    Outer as Outer,
    cross as cross,
    curl as curl,
    divergence as divergence,
    dot as dot,
    gradient as gradient,
    outer as outer,
)
from .utils import Domain as Domain, common as common, fastgl as fastgl
