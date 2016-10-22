# Set of functions which Newton finds the roots of, or tries to.

import numpy as N

def norm(x):
    return N.linalg.norm(x)

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
        Df_x[:,i] = (f(x + v) - fx) / dx
    return Df_x

class myfunc(object):
    def __call__(self, x):
        return 3.0 * x - 6.0

    def Jacobian(self, x):
        return N.matrix("3.0")

def f(x):
    return 3.0 * x - 6.0

def Df(x):
    return N.matrix("3.0")


class Exponential(object):
    def __init__(self,a=1.,b=1.,c=-1.):
        self._a = a
        self._b = b
        self._c = c
    def __repr__(self):
        return 'Exponential of the form y = a * exp(b*x) + c'
    def __call__(self,x):
        return self.f(x)
    def f(self,x):
        return self._a*N.exp(x*self._b) + self._c
    def Jacobian(self,x):
        return N.matrix(self._a*self._b*N.exp(x*self._b))
 
class Polynomial(object):
    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))
     
    def f(self,x): #original implementation seemed to have a sign error
        # ans = self._coeffs[0]
        #for c in self._coeffs[1:]:
        #    ans = x*ans + c
        ans = 0
        for n in range(0,len(self._coeffs)):
            ans+=self._coeffs[n] * x**n
        return ans
      
    def __call__(self, x):
        return self.f(x)

    def Jacobian(self,x):
        ans = 0
        for n in range(1,len(self._coeffs)):
            ans += n*self._coeffs[n] * x**(n-1)
        return N.matrix(ans)

class Polynomial2D(object): #hardcoded coefficients for now.
    """ y1 = (x1 + 10)(x2 - 10)
      y2 = (x1 + 1)(x2 - 100)."""
    def __init__(self):
        self._n = 2 #dimension
  
    def f(self,x):
        n = len(x)
        if n != self._n:
            raise ValueError('Size of input array x is '+str(len(x))+' but should be '+str(self._n))
        fx = N.mat(N.zeros((self._n,1)))
        fx[0,0] = (x[0]+10.)*(x[1]-10.)
        fx[1,0] = (x[0]+1)*(x[1]-100)
        return fx
  
    def __call__(self,x):
        return self.f(x)
  
    def Jacobian(self,x):
        Df = N.mat(N.zeros((self._n,self._n)))
        Df[0,0] = (x[1]-10)
        Df[0,1] = (x[0]+10) 
        Df[1,0] = x[1]-100 
        Df[1,1] = x[0]+1 
        return Df
  


class Polynomial2DWrongJacobian(object): #Computes the wrong analytical Jacobian
    """ y1 = (x1 + 10)(x2 - 10)
      y2 = (x1 + 1)(x2 - 100)."""
    def __init__(self):
        self._n = 2 #dimension
  
    def f(self,x):
        n = len(x)
        if n != self._n:
            raise ValueError('Size of input array x is '+str(len(x))+' but should be '+str(self._n))
        fx = N.mat(N.zeros((self._n,1)))
        fx[0,0] = (x[0]+10.)*(x[1]-10.)
        fx[1,0] = (x[0]+1)*(x[1]-100)
        return fx
  
    def __call__(self,x):
        return self.f(x)
  
    def Jacobian(self,x):
        Df = N.mat(N.zeros((self._n,self._n)))
        Df[0,0] = (x[1]-10)*1.1 #wrong, times 1.1 factor
        Df[0,1] = (x[0]+10) 
        Df[1,0] = x[1]-100 
        Df[1,1] = x[0]+1 - .4 #wrong, subtracted .4 
        return Df
  

  
  
  
  



