from .common import (
    Domain as Domain,
    diff as diff,
    diffx as diffx,
    fromdense as fromdense,
    jacn as jacn,
    lambdify as lambdify,
    matmat as matmat,
    reverse_dict as reverse_dict,
    tosparse as tosparse,
    ulp as ulp,
)
from .fastgl import leggauss as leggauss
from .sympy_factoring import (
    drop_time_argument as drop_time_argument,
    split_linear_nonlinear_terms as split_linear_nonlinear_terms,
    split_time_derivative_terms as split_time_derivative_terms,
)
