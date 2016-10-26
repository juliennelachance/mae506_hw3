#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        # Test Jacobian is accurate in one dimension.
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        # Test Jacobian is accurate in higher dimension (2). 
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    def testPolyJacobian(self):
        # p(x) = x^2 + 2x + 3 ; checks analytic Jacobian against approx Jacobian
        x0 = 2.0
        dx = 1.e-6
        f = F.Polynomial([1, 2, 3])
        analyDf_x = f.analyticJac(x0)
        approxDf_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(analyDf_x.shape, approxDf_x.shape)
        self.assertAlmostEqual(approxDf_x, analyDf_x, places=4)

    def testExponential(self):
        # f(x) = 1*exp(2*x) + 3
        fexp = F.Exponential(1, 2, 3)
        for x in N.linspace(-2,2,11):
            self.assertEqual(fexp(x), N.exp(2.0*x)+3)

    def testExpJacobian(self):
        # f(x) = 1*exp(2*x) + 3 ; checks analytic Jacobian against approx Jacobian
        x0 = 2.0
        dx = 1.e-6
        f = F.Exponential(1, 2, 3)
        analyDf_x = f.analyticJac(x0)
        approxDf_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(analyDf_x.shape, approxDf_x.shape)
        self.assertAlmostEqual(approxDf_x, analyDf_x, places=3)

    def testSineCosine(self):
        # sc(x) = sin(2x) + 3*cos(4x)
        sc = F.SineCosine(1, 2, 3, 4)
        for x in N.linspace(-2,2,11):
            self.assertEqual(sc(x), N.sin(2.0*x) + 3*N.cos(4.0*x))

    def testSinCosJacobian(self):
        # sc(x) = sin(2x) + 3*cos(4x) ; checks analytic Jacobian against approx Jacobian
        x0 = 2.0
        dx = 1.e-6
        f = F.SineCosine(1, 2, 3, 4)
        analyDf_x = f.analyticJac(x0)
        approxDf_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(analyDf_x.shape, approxDf_x.shape)
        self.assertAlmostEqual(approxDf_x, analyDf_x, places=4)

    def testTwoDJacobian(self):
        # Ensure we can calculate higher-dimension Jacobians. 
        x0 = N.mat("4.; 5.")
        dx = 1.e-6
        f = F.twoD_PolynomialFn()
        analyDf_x = f.analyticJac(x0)
        approxDf_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(analyDf_x.shape, approxDf_x.shape)
        N.testing.assert_array_almost_equal(approxDf_x, analyDf_x, decimal=4)


if __name__ == '__main__':
    unittest.main()



