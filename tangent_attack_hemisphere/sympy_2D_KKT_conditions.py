import json
import sympy
from sympy.vector import matrix_to_vector
from sympy import *
from sympy.solvers import solve
from sympy import Symbol,symbols
from sympy.matrices.dense import matrix_multiply_elementwise

w1, w2 = symbols('w1 w2',real = True)
w = Transpose(Matrix([[w1,w2]]))  # 列向量
a1,a2= symbols('a1 a2',real = True)
a = Transpose(Matrix([[a1, a2]]))
c1,c2 = symbols("c1 c2",real = True)
c = Transpose(Matrix([[c1,c2]]))
x1, x2= symbols("x1 x2",real = True)
x = Transpose(Matrix([[x1,x2]]))
lmbd, mu, R = symbols("lmdb mu R",positive=True,real=True)
#
# L = -(x1-a1) * w1 - (x2-a2) * w2 + lmbd * ((x1-a1) *(x1-c1) +(x2-a2) * (x2-c2)) + mu * ((x1-a1) ** 2 + (x2-a2) ** 2 - R**2)
square_x_a = matrix_multiply_elementwise(x-a,x-a)
L = -Transpose(x-a) * w + lmbd * (Transpose(x-a) * (x-c)) + mu * (Transpose(square_x_a) * square_x_a - R**2)
#
# equations = [diff(L, x1),diff(L, x2),
#              # -w2 + 2 * lmbd * x2 - lmbd * (a2 + c2) + 2 * mu * (x2-a2),
#              (x1-a1) * (x1-c1) + (x2-a2) * (x2-c2) ,
#              (x1-a1) ** 2 + (x2-a2) ** 2  - R**2 ]
# equations = [diff(L, x1)]
# x1 = (w1 + lmbd * (a1 + c1) + 2 * mu * a1) / (2 * (lmbd + mu))
# x2 = (w2 + lmbd * (a2 + c2) + 2 * mu * a2) / (2 * (lmbd + mu))
# x3 = (w3 + lmbd * (a3 + c3) + 2 * mu * a3) / (2 * (lmbd + mu))
#
# equations = [diff(L,x),
#              (x1-a1) * (x1-c1) + (x2-a2) * (x2-c2),
#              (x1-a1) ** 2 + (x2-a2) ** 2 - R**2 ]
equations = [diff(L,x),
             Transpose(x-a) * (x-c),
             Transpose(square_x_a) * square_x_a - R ** 2]
result = solve(equations,x,lmbd,mu,dict=True)
print(result)

