from scipy.optimize import fsolve
import numpy as np
def GRN(X):
    x = X[0]
    y = X[1]
    a1 = 1.
    a2 = 1.
    b1 = 0.2
    b2 = 0.2
    k = 1.1
    n = 2
    s = 0.5
    scale = 10
    return [a1*(x/scale)**n/(s**n+(x/scale)**n)+b1*s**n/(s**n+(y/scale)**n)-k*(x/scale),
            a2 * (y/scale) ** n / (s ** n + (y/scale) ** n) + b2 * s ** n / (s ** n + (x/scale) ** n) - k * (y/scale)]

for i in range(200):
    X0 = np.random.normal(0,10,[2])
    result = fsolve(GRN, X0)
    print(result)

# from sympy import symbols, Eq, solve, nsolve
#
# x, y = symbols('x y')
# a1 = 1.
# a2 = 1.
# b1 = 0.2
# b2 = 0.2
# k = 1.1
# n = 2
# s = 0.5
# eqs = [Eq(a1*x**n/(s**n+x**n)+b1*s**n/(s**n+y**n)-k*x, 0),
#        Eq(a2 * y ** n / (s ** n + y ** n) + b2 * s ** n / (s ** n + x ** n) - k * y, 0)]
#
# print(solve(eqs, [x, y]))
#
