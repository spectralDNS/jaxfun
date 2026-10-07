import jax.numpy as jnp
import numpy as np
import pytest

from jaxfun.galerkin import Chebyshev, Legendre, MeshKind
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.utils.common import Domain, ulp


def test_orthogonal_mapping_and_uniform_mesh():
    L = Legendre.Legendre(5, domain=Domain(-2, 3))
    x = L.system.x
    expr = x**3 + 2 * x + 1
    ref = L.map_expr_reference_domain(expr)
    tru = L.map_expr_true_domain(ref)
    # Instead of symbolic simplify (CoordSys interference), verify numerically
    pts = jnp.linspace(-2.0, 3.0, 5)
    symx = L.system.base_scalars()[0]
    for p in pts:
        val_orig = p**3 + 2 * p + 1
        val_mapped = tru.subs(symx, p)
        assert abs(float(val_mapped) - val_orig) < ulp(100.0)
    # numeric mapping arrays
    pts = jnp.linspace(-2.0, 3.0, 6)
    Xref = jnp.array([L.map_reference_domain(p) for p in pts])
    Xtrue = jnp.array([L.map_true_domain(X) for X in Xref])
    assert jnp.allclose(Xtrue, pts)
    # uniform mesh branch
    mesh_uniform = L.mesh(kind=MeshKind.UNIFORM, N=7)
    assert mesh_uniform.shape[0] == 7
    assert jnp.allclose(mesh_uniform, jnp.linspace(-2.0, 3.0, 7))


def test_chebyshev_uniform_mesh_and_reference_roundtrip():
    C = Chebyshev.Chebyshev(6, domain=Domain(-3, 1))
    pts = jnp.linspace(-3.0, 1.0, 4)
    Xref = jnp.array([C.map_reference_domain(p) for p in pts])
    Xtrue = jnp.array([C.map_true_domain(X) for X in Xref])
    assert jnp.allclose(Xtrue, pts)
    umesh = C.mesh(kind=MeshKind.UNIFORM, N=5)
    assert umesh.shape[0] == 5
    assert jnp.allclose(umesh, jnp.linspace(-3.0, 1.0, 5))


@pytest.mark.parametrize("n", [jnp.arange(6, 9)[0], np.int64(6)])
def test_integer_scalar_N_is_coerced_to_int(n):
    C = Chebyshev.Chebyshev(n)
    assert type(C.N) is int and C.num_dofs == 6
    D = FunctionSpace(n, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 0}})
    assert type(D.N) is int and D.num_dofs == 4


def test_non_integer_N_is_rejected():
    with pytest.raises(TypeError):
        Chebyshev.Chebyshev(6.0)  # ty: ignore[invalid-argument-type]
