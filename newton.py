# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import functions as F

class Newton(object):
    def __init__(self, f, Df= None, maxRad = None, tol=1.e-6, maxiter=20, dx=1.e-6, doConvTest=0):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian"""
        self._f = f
        self._Df = Df
        self._maxRad = maxRad
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self._doConvTest = doConvTest

    def solve(self, x0):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
        for i in xrange(self._maxiter):
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            else:
                x = self.step(x, fx)
            if self._maxRad:
                if N.linalg.norm(x-x0) > self._maxRad:
                    raise RuntimeError("Approx. root does not lie within defined radius")
        if self._doConvTest:
            if N.linalg.norm(fx) > self._tol:
                raise RuntimeError("Method failed to converge after max no. of iterations")
        return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        if self._Df is None:
            Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        else:
            Df_x = self._Df
        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        return x - h
