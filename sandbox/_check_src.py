import jax; jax.config.update('jax_enable_x64', True)
import inspect
from jaxfun.galerkin.tensorproductspace import tpmats_to_kron
src = inspect.getsource(tpmats_to_kron)
# Print the _to_dia portion
start = src.find('def _to_dia')
print(src[start:start+200] if start >= 0 else 'NOT FOUND')
