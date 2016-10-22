import functions as F
import unittest
import numpy as N

class TestFunctions(unittest.TestCase):
    def testJacobian1(self):
        for slope in range(-3,4):
            for x0 in range(-2,3):
                x0 = N.matrix(x0)
                h = 1e-6
                f = lambda x: slope * x
                Df_x = F.ApproximateJacobian(f, x0, h)
                self.assertEqual(Df_x.shape, (1,1))
                self.assertAlmostEqual(Df_x, slope)

    def testJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        f = lambda x : A * x
        x = N.matrix("5; 6")
        dx = 1e-6
        Df_x = F.ApproximateJacobian(f, x, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testJacobianArray(self):
        A = N.array([[1., 2.],[3., 4.]])
        f = lambda x : N.dot(A, x)
        x = N.array([[5],[6]])
        dx = 1e-6
        Df_x = F.ApproximateJacobian(f, x, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testExponentialf(self):
        myexp = F.Exponential(a=2.,b=-3.,c=-1.)
        x=1.
        answer = -0.900425863264272 #from another source, matlab
        self.assertAlmostEqual(answer,myexp.f(x))

    def testPolynomial(self):
        #Test polynomial y = (2*x+1)*(x-2)*(x+4) = -8 -14*x +5*x**2 +2x**3
        #First test that f(x) works correctly.
        coeffs = [-8., -14., 5.,2.]
        mypoly = F.Polynomial(coeffs=coeffs)
        xs = [-1.,0.,50.]
        fxs = [9.,-8.,261792.]
        for (x,fx) in zip(xs,fxs):
            self.assertAlmostEqual(fx,mypoly.f(x))
        #Now test Jacobian, true values (from matlab)
        #This was moved to the testAllAnalyticalJacobians function.
        """
        dfxs = [-18.,-14. ,15486.]
        for (x,dfx) in zip(xs,dfxs):
            self.assertAlmostEqual(dfx,mypoly.Jacobian(x)[0,0],places=6)
        for x in xs:
            #adjusted the precision, our approx is not accurate enough by default.
            self.assertAlmostEqual(mypoly.Jacobian(x)[0,0],F.ApproximateJacobian(mypoly.f,x)[0,0],places=3)
        """

    def testPolynomial2Df(self):
        #Test 3 points, x,f(x) pairs.
        myPoly2D = F.Polynomial2D()
        xs = [N.mat("-5.;10."),N.mat("0.; 0."),N.mat("100.;-2")]
        fxs = [N.mat("0.; 360."),N.mat("-100.;-100"),N.mat("-1320.;-10302.")]
        for (x,fx) in zip(xs,fxs):
            N.testing.assert_array_almost_equal(myPoly2D.f(x),fx,decimal=4)
    #Tests the exact and approximate jacobians for 2D polynomial
    #This function is now part of testAllAnalyticalJacobians below.
    """
    def testPolynomial2DJacobian(self):
        dx = 1e-6
        myPoly2D = F.Polynomial2D()
        xs = [N.mat("2. ; 3."),N.mat("-4.;5.")]
        for x in xs:
            DfAnalytical = myPoly2D.Jacobian(x)
            DfNumerical = F.ApproximateJacobian(myPoly2D.f,x,dx=dx)
            #lowered tolerance from default in assertion below.
            N.testing.assert_array_almost_equal(DfAnalytical,DfNumerical,decimal=4)
    """

    def testAllAnalyticalJacobians(self):
        #This is the simple way to add more tests of analytical Jacobians.
        #Creates a tuple with the new function and tests a list of points for each
        # new case.
        #Simply add a new line following the pattern of the others.

        dx = 1e-6 #used for calculating the approximate jacobian.
        mydecimal = 4
        caseList = []
        #decimal is the number of decimals that should match in comparisons below
        caseList.append((F.Polynomial2D(),[N.mat("2. ; 3."),N.mat("-4.;5.")],dx,mydecimal))
        caseList.append((F.Exponential(a=1,b=1,c=-3),[N.mat(2.),N.mat(-5),N.mat(0.01)],dx,mydecimal))
        caseList.append((F.Polynomial( [-8., -14., 5.,2.]),[-1.,0.,50.],dx,3))
        # ...
        for (func,xList,dx,mydecimal) in caseList:
            for x in xList:
                DfAnalytical = func.Jacobian(x)
                DfNumerical = F.ApproximateJacobian(func.f,x,dx=dx)
                N.testing.assert_array_almost_equal(DfAnalytical,DfNumerical,decimal=mydecimal)





if __name__ == '__main__':
  unittest.main()



