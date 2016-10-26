import numpy as N

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/dx
    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans

    def analyticJac(self, x):
        ans = 0
        n = len(self._coeffs)
        for i in range(0,n-1):
            ans += (n-i-1)*self._coeffs[i] * x**((n-i-2)) 
        return N.matrix(ans)

    def __call__(self, x):
        return self.f(x)

class Exponential(object):
    """Callable exponential object.

    Example usage: to construct the exponential function ex(x) = a*e^(b*x) + c,
    and evaluate ex(5):

    ex = Exponential([1, 2, 3])
    ex(5)"""

    def __init__(self, a=0, b=1, c=0):
        self._a = a
        self._b = b
        self._c = c

    def __repr__(self):
        return "Exponential f(x) = a*exp(b*x) + c" 

    def f(self,x):
        ans = self._a*N.exp(self._b * x) + self._c
        return ans

    def analyticJac(self, x):
        ans = self._a*self._b*N.exp(self._b * x) 
        return N.matrix(ans)

    def __call__(self, x):
        return self.f(x)

class SineCosine(object):
    """Callable sine/cosine object.

    Example usage: to construct the function sc(x) = a*sin(b*x) + c*cos(d*x),
    and evaluate sc(5):

    sc = SineCosine([1, 2, 3, 4])
    sc(5)"""

    def __init__(self, a=0, b=1, c=0, d=1):
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def __repr__(self):
        return "SineCosine f(x) = a*sin(b*x) + c*cos(d*x)"

    def f(self,x):
        ans = self._a * N.sin(self._b * x) + self._c*N.cos(self._d * x)
        return ans

    def analyticJac(self, x):
        ans = self._a*self._b*N.cos(self._b * x) - self._c*self._d*N.sin(self._d * x)
        return N.matrix(ans)

    def __call__(self, x):
        return self.f(x)

class twoD_PolynomialFn():

    def __repr__(self):
        return "2D Polynomial f(x) = [x1*x2 ; x1+x2]"

    def f(self, x):
        twoD_f = N.mat(N.zeros((2,1)))
        twoD_f[0,0] = x[0] + x[1] - 7
        twoD_f[1,0] = x[0] + 2*x[1] - 11
        return twoD_f

    def analyticJac(self, x):
        twoD_Jac = N.mat(N.zeros((2,2)))
        twoD_Jac[0,0] = 1
        twoD_Jac[0,1] = 1
        twoD_Jac[1,0] = 1
        twoD_Jac[1,1] = 2
        return twoD_Jac

    def __call__(self, x):
        return self.f(x)











