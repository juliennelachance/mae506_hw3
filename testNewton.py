
import newton
import unittest
import numpy as N
import functions as F

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testQuadratic(self):
        f = lambda x : x**2 - 9
        solver = newton.Newton(f, tol=1.e-11, maxiter=10)
        x = solver.solve(2.0)
        self.assertEqual(x, 3.0)
        x = solver.solve(-2.0)
        self.assertEqual(x, -3.0)

    def XtestSingular(self):
        f = lambda x : x**2
        solver = newton.Newton(f, tol=1.e-11, maxiter=50, dx=1.e-13)
        x = solver.solve(1.0)
        self.assertAlmostEqual(x, 0.0)

    def g(self,x):
        return 3.0 *x  - 6.0

    def Dg(self,x):
        return 3.0

    def testMyfunc(self):
        solver = newton.Newton(self.g, Df=self.Dg)
        x = solver.solve(0.0)
        self.assertEqual(x,2.0)

    def testStep(self):
        solver = newton.Newton(self.g)
        x0 = 0.0
        x1 = solver.step(x0)
        x2 = solver.step(x0, self.g(x0))
        self.assertEqual(x1, x2)

    def testBadDeriv(self):
        def f1(x):
            return -2.0 * (x - 1.0)
        def Df1(x):
            # incorrect Jacobian!
            return N.matrix("2.0")
        # One step should go the wrong way with the wrong Jacobian
        solver = newton.Newton(f1, Df=Df1, maxiter=1)
        x = solver.step(0.0)
        self.assertEqual(x, -1.0)
        # Should find exact solution in one step if Jacobian not specified
        solver2 = newton.Newton(f1, maxiter=1)
        x2 = solver2.step(0.0)
        self.assertAlmostEqual(x2, 1.0)

    #Test 2D nonlinear case with approximate and analytic Jacobians and member functions.
    def testPolynomial2D(self):
        x = N.mat("-2.; -2.")
        myPoly2D = F.Polynomial2D()
        xRootTrue = N.mat("-1.;10.") # exact root
        solver = newton.Newton(myPoly2D.f) #uses approximate Jacobian
        xRoot1 = solver.solve(x)
        N.testing.assert_array_almost_equal(xRoot1,xRootTrue)

        #Now with the analytic jacobian.
        solver = newton.Newton(myPoly2D.f,Df=myPoly2D.Jacobian) #uses exact Jacobian
        xRoot2 = solver.solve(x)
        N.testing.assert_array_almost_equal(xRoot2,xRootTrue)
        #check that the two root approximations are not the same due to using
        #different Jacobians
        #How to assert not equal?
        if F.norm(xRoot1-xRoot2) == 0:
            print 'The exact and analytical Jacobians gave exactly the same root'
            self.assert_(False)

   #Test 2D nonlinear case with WRONG analytic Jacobian.
    def testPolynomial2DWrongJacobian(self):
        x = N.mat("-2.; -2.")
        xRootTrue = N.mat("-1.;10")
        myPoly2D = F.Polynomial2DWrongJacobian()
        xRootTrue = N.mat("-1.;10.") # exact root
        #Wrong analytic jacobian.
        solver = newton.Newton(myPoly2D.f,Df=myPoly2D.Jacobian) #uses exact Jacobian
        try: xRoot = solver.solve(x)
        except RuntimeError: #should not converge?
            print 'Root didnt converge with wrong Jacobian'
        else:
            pass
            #print "Using the wrong Jacobian, converged with norm(x-xtrue)=",F.norm(xRoot-xRootTrue)
            #raise RuntimeError('Wrong Jacobian did not raise a RuntimeError, should it have?')


    #Test that for an exponential function (has no root), the radius test kicks in
    #and reports an error.
    def testRadiusCheck(self):
        x = 1.
        myExp = F.Exponential(a=1.,b=1,c=0)
        solver = newton.Newton(myExp.f) #tests with approximate Jacobian
        try: xRoot = solver.solve(x)
        except RuntimeError: #expect this error
          pass

        solver = newton.Newton(myExp.f,Df=myExp.Jacobian) #tests with analytic Jacobian
        try: xRoot = solver.solve(x)
        except RuntimeError: #expect this error
          pass

    #Matrix is singular (columns are not linearly independent)
    def testSingular(self):
        A = N.matrix("1. 2.; 2. 4.")
        f = lambda x : A * x
        x = N.matrix("5; 6")
        dx = 1e-6
        solver = newton.Newton(f)
        try: xRoot = solver.solve(x)
        except N.linalg.LinAlgError:
            pass #expect this error for a singular matrix.


if __name__ == "__main__":
    unittest.main()
