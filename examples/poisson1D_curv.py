r"""
Solve Poisson's equation on a curved line in 2D or 3D space.

Define a position vector, `rv(t)`, as::

    rv(t) = x(t)i + y(t)j + z(t)k,

where i, j, k are the Cartesian unit vectors and t is found
in some suitable interval, e.g., [0, 1] or [0, 2\pi]. Note that
the position vector describes a, possibly curved, 1D domain.

Solve::

    -div(grad(u(t))) = f(t), for t in interval

using curvilinear coordinates.

"""
import sys
import os
import sympy as sp
import numpy as np
from sympy.plotting import plot3d_parametric_line
import jax.numpy as jnp 
from jaxfun.utils.common import lambdify, ulp
from jaxfun.Legendre import Legendre as space
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.operators import Grad, Div, Dot
from jaxfun.Basespace import n
from jaxfun.coordinates import get_CoordSys

t = sp.Symbol('t', real=True)
rv = (sp.sin(2*sp.pi*t), sp.cos(2*sp.pi*t), 2*t)

N = 50
bcs = {'left': {'D': 0}, 'right': {'D': 0}}
coors = get_CoordSys("C", sp.Lambda((t,), rv))
D = FunctionSpace(N, space, bcs, scaling=n+1, system=coors, name='D', fun_str='phi')
v = TestFunction(D)
u = TrialFunction(D)

# Method of manufactured solution
t = D.system.t # use the same coordinate as u and v

ue = sp.sin(4*sp.pi*t)
b = inner(-v*Div(Grad(ue)))
#b = inner(-v*sp.Derivative(ue, t, 2))
A = inner(-v*Div(Grad(u)), sparse=True)

u_hat = jnp.linalg.solve(A.todense(), b)

xj = D.orthogonal.quad_points_and_weights()[0]
uj = D.evaluate(xj, u_hat)
uq = lambdify(t, ue)(xj)
error = np.linalg.norm(uj-uq)
if 'pytest' in os.environ:
    assert error < 1000*ulp(1)
    sys.exit(1)

print(f'poisson1D_curv l2 error = {error:2.6e}')

# For plotting, transform base_scalars to pure sympy symbols
x, y, z = D.system.expr_base_scalar_to_psi(coors.rv)
t = t.to_symbol()
ue = D.system.expr_base_scalar_to_psi(ue)
ax = plot3d_parametric_line((x, y, z, (t, -1, 1)), (x, y, z+ue, (t, -1, 1)))
