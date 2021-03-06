import json
import sympy
from sympy import *
from sympy.solvers import solve
from sympy import Symbol,symbols
w1, w2,w3 = symbols('w1 w2 w3',real = True)
a1,a2,a3= symbols('a1 a2 a3',real = True)
c1,c2,c3 = symbols("c1 c2 c3",real = True)
x1, x2,x3= symbols("x1 x2 x3",real = True)
lmbd, mu, R = symbols("lmdb mu R",positive=True,real=True)
#
L = -(x1-a1) * w1 - (x2-a2) * w2 - (x3-a3) * w3 + lmbd * ((x1-a1) *(x1-c1) +(x2-a2) * (x2-c2) + (x3-a3) * (x3-c3) ) + mu * ((x1-a1) ** 2 + (x2-a2) ** 2 + (x3-a3) ** 2 - R**2)
#
# equations = [diff(L, x1),diff(L, x2),
#              # -w2 + 2 * lmbd * x2 - lmbd * (a2 + c2) + 2 * mu * (x2-a2),
#              (x1-a1) * (x1-c1) + (x2-a2) * (x2-c2) ,
#              (x1-a1) ** 2 + (x2-a2) ** 2  - R**2 ]
# equations = [diff(L, x1)]
# x1 = (w1 + lmbd * (a1 + c1) + 2 * mu * a1) / (2 * (lmbd + mu))
# x2 = (w2 + lmbd * (a2 + c2) + 2 * mu * a2) / (2 * (lmbd + mu))
# x3 = (w3 + lmbd * (a3 + c3) + 2 * mu * a3) / (2 * (lmbd + mu))
equations = [diff(L,x1), diff(L,x2), diff(L,x3),
             (x1-a1) * (x1-c1) + (x2-a2) * (x2-c2) +(x3-a3) * (x3-c3) ,
             (x1-a1) ** 2 + (x2-a2) ** 2 + (x3-a3) ** 2 - R**2 ]
result = solve(equations,x1,x2,x3, lmbd,mu,dict=True)
print(result)

# {x1: (-R**2*a1 + R**2*c1 + R*a2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) - R*c2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + a1**3 - 2*a1**2*c1 + a1*a2**2 - 2*a1*a2*c2 + a1*c1**2 + a1*c2**2)/(a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2), x2: (-R**2*a2 + R**2*c2 - R*a1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + R*c1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + a1**2*a2 - 2*a1*a2*c1 + a2**3 - 2*a2**2*c2 + a2*c1**2 + a2*c2**2)/(a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2), lmdb: (-R**2*a1*w1 - R**2*a2*w2 + R**2*c1*w1 + R**2*c2*w2 - R*a1*w2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + R*a2*w1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + R*c1*w2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) - R*c2*w1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + a1**3*w1 + a1**2*a2*w2 - 3*a1**2*c1*w1 - a1**2*c2*w2 + a1*a2**2*w1 - 2*a1*a2*c1*w2 - 2*a1*a2*c2*w1 + 3*a1*c1**2*w1 + 2*a1*c1*c2*w2 + a1*c2**2*w1 + a2**3*w2 - a2**2*c1*w1 - 3*a2**2*c2*w2 + a2*c1**2*w2 + 2*a2*c1*c2*w1 + 3*a2*c2**2*w2 - c1**3*w1 - c1**2*c2*w2 - c1*c2**2*w1 - c2**3*w2)/(-R**2*a1**2 + 2*R**2*a1*c1 - R**2*a2**2 + 2*R**2*a2*c2 - R**2*c1**2 - R**2*c2**2 + a1**4 - 4*a1**3*c1 + 2*a1**2*a2**2 - 4*a1**2*a2*c2 + 6*a1**2*c1**2 + 2*a1**2*c2**2 - 4*a1*a2**2*c1 + 8*a1*a2*c1*c2 - 4*a1*c1**3 - 4*a1*c1*c2**2 + a2**4 - 4*a2**3*c2 + 2*a2**2*c1**2 + 6*a2**2*c2**2 - 4*a2*c1**2*c2 - 4*a2*c2**3 + c1**4 + 2*c1**2*c2**2 + c2**4), mu: -(a1*w1 + a2*w2 - c1*w1 - c2*w2)/(a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + (-a1*w2 + a2*w1 + c1*w2 - c2*w1)*(-2*R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2)*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2)/(2*R*(-R**2*a1**2 + 2*R**2*a1*c1 - R**2*a2**2 + 2*R**2*a2*c2 - R**2*c1**2 - R**2*c2**2 + a1**4 - 4*a1**3*c1 + 2*a1**2*a2**2 - 4*a1**2*a2*c2 + 6*a1**2*c1**2 + 2*a1**2*c2**2 - 4*a1*a2**2*c1 + 8*a1*a2*c1*c2 - 4*a1*c1**3 - 4*a1*c1*c2**2 + a2**4 - 4*a2**3*c2 + 2*a2**2*c1**2 + 6*a2**2*c2**2 - 4*a2*c1**2*c2 - 4*a2*c2**3 + c1**4 + 2*c1**2*c2**2 + c2**4))}, {x1: (-R**2*a1 + R**2*c1 - R*a2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + R*c2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + a1**3 - 2*a1**2*c1 + a1*a2**2 - 2*a1*a2*c2 + a1*c1**2 + a1*c2**2)/(a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2), x2: (-R**2*a2 + R**2*c2 + R*a1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) - R*c1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + a1**2*a2 - 2*a1*a2*c1 + a2**3 - 2*a2**2*c2 + a2*c1**2 + a2*c2**2)/(a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2), lmdb: (-R**2*a1*w1 - R**2*a2*w2 + R**2*c1*w1 + R**2*c2*w2 + R*a1*w2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) - R*a2*w1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) - R*c1*w2*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + R*c2*w1*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) + a1**3*w1 + a1**2*a2*w2 - 3*a1**2*c1*w1 - a1**2*c2*w2 + a1*a2**2*w1 - 2*a1*a2*c1*w2 - 2*a1*a2*c2*w1 + 3*a1*c1**2*w1 + 2*a1*c1*c2*w2 + a1*c2**2*w1 + a2**3*w2 - a2**2*c1*w1 - 3*a2**2*c2*w2 + a2*c1**2*w2 + 2*a2*c1*c2*w1 + 3*a2*c2**2*w2 - c1**3*w1 - c1**2*c2*w2 - c1*c2**2*w1 - c2**3*w2)/(-R**2*a1**2 + 2*R**2*a1*c1 - R**2*a2**2 + 2*R**2*a2*c2 - R**2*c1**2 - R**2*c2**2 + a1**4 - 4*a1**3*c1 + 2*a1**2*a2**2 - 4*a1**2*a2*c2 + 6*a1**2*c1**2 + 2*a1**2*c2**2 - 4*a1*a2**2*c1 + 8*a1*a2*c1*c2 - 4*a1*c1**3 - 4*a1*c1*c2**2 + a2**4 - 4*a2**3*c2 + 2*a2**2*c1**2 + 6*a2**2*c2**2 - 4*a2*c1**2*c2 - 4*a2*c2**3 + c1**4 + 2*c1**2*c2**2 + c2**4), mu: -(a1*w1 + a2*w2 - c1*w1 - c2*w2)/(a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2) - (-a1*w2 + a2*w1 + c1*w2 - c2*w1)*(-2*R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2)*sqrt(-R**2 + a1**2 - 2*a1*c1 + a2**2 - 2*a2*c2 + c1**2 + c2**2)/(2*R*(-R**2*a1**2 + 2*R**2*a1*c1 - R**2*a2**2 + 2*R**2*a2*c2 - R**2*c1**2 - R**2*c2**2 + a1**4 - 4*a1**3*c1 + 2*a1**2*a2**2 - 4*a1**2*a2*c2 + 6*a1**2*c1**2 + 2*a1**2*c2**2 - 4*a1*a2**2*c1 + 8*a1*a2*c1*c2 - 4*a1*c1**3 - 4*a1*c1*c2**2 + a2**4 - 4*a2**3*c2 + 2*a2**2*c1**2 + 6*a2**2*c2**2 - 4*a2*c1**2*c2 - 4*a2*c2**3 + c1**4 + 2*c1**2*c2**2 + c2**4))}
