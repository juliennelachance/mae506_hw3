# mae506_hw3
Assignment 3: Root finding and automated testing

## What was modified: 
The provided files newton.py and functions.py have been debugged. 
The Newton class has been updated to allow the user to specify an analytic Jacobian. It has also been modified
   to allow the user to test whether or not the method has converged in the allotted number of iterations. 
   This was implemented as a new argument in order to allow the provided tests to pass without modification.
A condition was added to ensure that the approximated root lies within a max radius of the initial guess (by Euclidean norm).

GitHub was used to manage code.

## Description of added tests:
The following additional tests for newton.py and functions.py were created in testNewton.py:

#testLinear: 
Solves a simple linear function and checks method output with known solution.

#testQuadratic: 
Solves a simple quadratic function and checks method output with known solution.

#testTwoDSystem:
Solves a simple 2D polynomial function and checks method output with known solution. The purpose 
was to ensure that the method handles higher-order systems appropriately. 

#testExponentialS:
Solves a simple exponential function and checks method output with known solution.

#testSinCosS:
Solves a simple function consisting of a combination of sine and cosine terms and checks method output with known solution.

#testOneStep:
Ensures a single step of the Newton method performs as it should.

#testLinearConvergence:
An extra feature was included in the Newton class to allow the user to check whether or not the method converges
within the max number of iterations. First the method runs a tests which should converge within the max number of 
iterations, and we check that no error is thrown. Then we test a case which should not converge under these conditions
and check that the RuntimeError is thrown. 

#testMethodUsesAnalytic:
Creates a new (incorrect) Jacobian for an input function and feeds this into the Newton method. The test ensures that 
the resulting solution matches an appropriate incorrect solution- which ensures that the method is using the input
analytic Jacobian. 

#testAnalyticJacobianSolutions:
This is a test to ensure that the solution that results from a the approximate Jacobian and a true Jacobian are reasonably
similar, while also ensuring that the approximate solution is sufficiently different from a solution resulting from a 
badly constructed analytic Jacobian. It is similar to tests of the approximate/analytic Jacobians in testFunction.py but
carried out to the solution values as well. 

#testMaxRadiusFeature:
We implemented a new feature to throw a RuntimeError if the user inputs a maximum radius value and the solution fails to 
fall within this radius (maxRad). This test ensures this functionality by creating a function which has no real solution
and assuring that the RuntimeError is thrown. 


##The following additional tests for newton.py and functions.py were created in testFunctions.py:

#testApproxJacobian1:
Tests a one-dimensional Jacobian by comparing approximate Jacobian to known analytic solution. 

#testApproxJacobian2:
Tests a two-dimensional Jacobian by comparing approximate Jacobian to known analytic solution. 

#testPolynomial:
Tests that the polynomial class creates a function which matches a true function over a range of x-values. 

#testPolyJacobian:
Tests that the approximate Jacobian is reasonably close to an analytic Jacobian for a fixed polynomial function. 

#testExponential:
Tests that the exponential class creates a function which matches a true function over a range of x-values. 

#testExpJacobian:
Tests that the approximate Jacobian is reasonably close to an analytic Jacobian for a fixed exponential function. 

#testSineCosine:
Tests that the sine/cosine class creates a function which matches a true function over a range of x-values. 

#testSinCosJacobian:
Tests that the approximate Jacobian is reasonably close to an analytic Jacobian for a fixed sine/cosine function. 

#testTwoDJacobian:
Tests that the approximate Jacobian is reasonably close to an analytic Jacobian for a fixed 2D polynomial function. 



## General functionality: 
The code base consists of the following functions:

testNewton.py: A testing suite which calls to newton.py and functions.py, and serves to ensure that newton.py
   works appropriately in a number of ways. 
testFunctions.py: Another testing suite like testNewton.py but one which ensures that functions.py is working
   correctly. 
newton.py: defines the Newton class and contains the functions to perform Newton's Method for root finding. 
   The method uses linear approximations to iterate towards an approximation of the root of an input function. 
functions.py: contains the function to approximate the Jacobian matrix which is used by newton.py. Also contains
   structures like the Polynomial class to help the user generate functions to use for testing newton.py. 
