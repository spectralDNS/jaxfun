import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import sparse as scipy_sparse

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.basespace import n
from jaxfun.coordinates import get_CoordSys
from jaxfun.Fourier import Fourier
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.Legendre import Legendre
from jaxfun.operators import Div, Grad
from jaxfun.tensorproductspace import TensorProduct, tpmats_to_scipy_kron
from jaxfun.utils.common import lambdify, ulp

M = 20
bcs = {"left": {"D": 0}, "right": {"D": 0}}
r, theta, z = sp.symbols("r,theta,z", real=True, positive=True)
C = get_CoordSys(
    "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
)
D0 = FunctionSpace(
    M, Legendre, bcs, scaling=n + 1, domain=(0, 1), name="D0", fun_str="psi"
)
D1 = FunctionSpace(M, Fourier, domain=(0, 2 * sp.pi), name="D1", fun_str="phi")
D2 = FunctionSpace(M, Fourier, name="F", fun_str="E")
T = TensorProduct((D0, D1, D2), system=C, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")
