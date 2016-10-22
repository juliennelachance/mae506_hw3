# newton - Newton-Raphson solver, with tests
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import unittest
import numpy.linalg #for matrix inversion and norms
import numpy.testing
import functions as F

class Newton(object):
    def __init__(self, f, Df=None, tol=1.e-6, maxiter=20, dx=1.e-6, \
                     maxRadius=None):
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self._Df = Df
        self._maxRadius = maxRadius

    def solve(self, x0):
        x = x0
        for i in xrange(self._maxiter):
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            else:
                x = self.step(x, fx)
            if self._maxRadius and N.linalg.norm(x-x0) > self._maxRadius:
                raise RuntimeError("Approximate root out of range")
        return x

    def step(self, x, fx=None):
        if fx is None:
            fx = self._f(x)
        if self._Df:
            Df_x = self._Df(x)
        else:
            Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        return x - h

