#!/usr/bin/env python

import newton
import unittest
import numpy as N
import functions as F

# todo
#Add any other tests you think are useful.
# README 


class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testQuadratic(self):
        # Ensure we can handle higher-order functions.
        f = lambda x : x**2 - x - 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(2.0)
        self.assertEqual(x, 3)

    def testTwoDSystem(self):
        # Ensure we can solve higher-dimension systems. 
        f = F.twoD_PolynomialFn()
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x0 = N.mat("1.; 1.")
        x_true = N.mat("3.; 4.")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, x_true)

    def testExponentialS(self):
        # Ensure that we find correct root of exponential function.
        # f(x) = 1*exp(-1*x) - 1 
        x0 = 2.0
        x_true = 0.0
        f = F.Exponential(1, -1, -1)
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(x0)
        self.assertAlmostEqual(x, x_true)

    def testSinCosS(self):
        # Ensure that we find correct root of function composed of sines/cosines.
        # sc(x) = 1.0*sin(1.0*x) + -1.0*cos(1.0*x) 
        x0 = -5.5
        x_true = -5.4977871437821381 # from WolframAlpha.com 
        f = F.SineCosine(1, 1, -1.0, 1)
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(x0)
        self.assertAlmostEqual(x, x_true)

    def testOneStep(self):
        # Ensure a single step performs as it should.
        f = lambda x : x**2 - 5.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=1)
        x = solver.solve(2.0)
        self.assertAlmostEqual(round(x,5), 2.25)

    def testLinearConvergence(self):
        # Tests functionality of extra feature to check when method does not converge
        # within max number of iterations. 
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=3, doConvTest=1)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)
        solver = newton.Newton(f, tol=1.e-15, maxiter=1, doConvTest=1)
        try: x = solver.solve(2.0)
        except RuntimeError:  # This should fail 
            pass

    def testMethodUsesAnalytic(self):
        # Ensures that the analytic Jacobian is actually the one used by the root finder
        # Use an incorrect Jacobian for the input function; check one step of solution
        f = F.twoD_PolynomialFn()  # 2D polynomial example
        Df_analy = N.mat(N.zeros((2,2)))  # Wrong Jacobian for input function
        Df_analy[0,0] = 2
        Df_analy[0,1] = 1
        Df_analy[1,0] = 1
        Df_analy[1,1] = 2

        solver = newton.Newton(f, Df = Df_analy, tol=1.e-15, maxiter=1)
        x0 = N.mat("1.; 1.")
        x = solver.solve(x0)
        x_fromDf = N.mat("1.66666667; 4.66666667")
        N.testing.assert_array_almost_equal(x, x_fromDf)

    def testAnalyticJacobianSolutions(self):
        f = F.twoD_PolynomialFn()  # 2D polynomial example
        x0 = N.mat("1.; 1.")
        Bad_Df = N.mat(N.zeros((2,2)))  # Incorrect Jacobian for input function
        Bad_Df[0,0] = 2
        Bad_Df[0,1] = 1
        Bad_Df[1,0] = 1
        Bad_Df[1,1] = 2
        Good_Df = N.mat(N.zeros((2,2)))  # Correct Jacobian for input function
        Good_Df[0,0] = 1
        Good_Df[0,1] = 1
        Good_Df[1,0] = 1
        Good_Df[1,1] = 2
        # Solve the system with approximate Jacobian:
        solver_approx = newton.Newton(f, tol=1.e-15, maxiter=10)
        x_approx = solver_approx.solve(x0)
        # Solve the system with analytic Jacobian:
        solver_goodAnaly = newton.Newton(f, Df = Good_Df, tol=1.e-15, maxiter=10)
        x_goodAnaly = solver_goodAnaly.solve(x0)
        # Check solutions approximately equal:
        self.assertEqual(x_approx.shape, x_goodAnaly.shape)
        N.testing.assert_array_almost_equal(x_approx, x_goodAnaly, decimal=4)
        # Now, solve the system with bad analytic Jacobian to show different results:
        solver_badAnaly = newton.Newton(f, Df = Bad_Df, tol=1.e-15, maxiter=10)
        x_badAnaly = solver_badAnaly.solve(x0)
        try: N.testing.assert_array_almost_equal(x_approx, x_badAnaly, decimal=4)
        except AssertionError: # Should fail
            pass

    def testMaxRadiusFeature(self):
        # Construct a function to test a case where solution lies outside of maximum radius (maxRad).
        # We choose a function with no real solution and ignore other errors from numpy:
        # p(x) = x^2 + 1 = 0 (pure imaginary roots)
        p = F.Polynomial([1, 2, 3])
        solver = newton.Newton(p, maxRad = 1.0, tol=1.e-15, maxiter=5)
        try: x = solver.solve(2.0)
        except RuntimeError: # We want this to fail.
          pass


if __name__ == "__main__":
    unittest.main()





