from jaxfun import *
import sympy as sp


def run_1D():
    L = Legendre.Legendre(8)
    u = TrialFunction(L, name='u')
    v = TestFunction(L, name='v')
    x = L.system.x

    a = [None]*10
    a[0] = Div(Grad(u))*v 
    a[1] = Dot(Grad(u), Grad(v))
    a[2] = x*a[0]
    a[3] = x*a[1]
    a[4] = (1+x)*a[0]
    a[5] = (1+x)*a[1]
    a[6] = Div(Grad(Div(Grad(u))))*v 
    a[7] = Div(Grad(u))*Div(Grad(v))
    a[8] = v*sp.diff(u, x, 1)
    a[9] = (1+x)*v*sp.diff(u, x, 2)
    
    for i in a:
        d = forms.split(i)
    return a

def run_2D():
    L = Legendre.Legendre(8)
    T = TensorProductSpace((L, L), name='T')
    u = TrialFunction(T, name='t0')
    v = TestFunction(T, name='t1')
    x, y = T.system.x, T.system.y

    b = [None]*11
    b[0] = Div(Grad(u))*v 
    b[1] = Dot(Grad(u), Grad(v))
    b[2] = x*b[0]
    b[3] = y*b[1]
    b[4] = (1+x)*b[0]
    b[5] = (1+y)*b[1]
    b[6] = Div(Grad(Div(Grad(u))))*v 
    b[7] = Div(Grad(u))*Div(Grad(v))
    b[8] = v*sp.diff(u, x, 1)
    b[9] = (1+x)*v*sp.diff(u, x, 2)
    b[10] = (1+x)*sp.diff(v, x, 1)*sp.diff(u, y, 2)

    for i in b:
        print(i)
        d = forms.split(i)
    return b

def run_cyl():
    r, theta = sp.symbols('r,theta', real=True)
    C = get_CoordSys('C', sp.Lambda((r, theta), (r*sp.cos(theta), r*sp.sin(theta))))
    L = Legendre.Legendre(8, system=C._parent)
    Q = TensorProductSpace((L, L), system=C, name='Q')
    u = TrialFunction(Q, name='q0')
    v = TestFunction(Q, name='q1')
    x, y = Q.system.x, Q.system.y

    c = [None]*7
    c[0] = Div(Grad(u))*v 
    c[1] = Dot(Grad(u), Grad(v))
    c[2] = sp.sqrt(x**2+y**2)*c[0]
    c[3] = Div(Grad(Div(Grad(u))))*v 
    c[4] = Div(Grad(u))*Div(Grad(v))
    c[5] = r*Div(Grad(u))*v
    c[6] = r*v

    for i in c:
        d = forms.split(i)
    return c

if __name__ == '__main__':
    a = run_1D()
    b = run_2D()
    c = run_cyl()