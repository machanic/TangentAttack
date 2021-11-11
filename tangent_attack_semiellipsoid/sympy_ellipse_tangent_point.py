import sympy
from sympy.vector import matrix_to_vector
from sympy import *
from sympy.solvers import solve
from sympy import Symbol,symbols
from sympy.matrices.dense import matrix_multiply_elementwise

L,S = symbols("L S",positive=True,real=True)
X0,Z0 = symbols("X0 Z0",real = True)
Xk,Zk = symbols("Xk Zk",real = True)
equations = [L**2 * Xk**2 + S**2 * Zk ** 2 - Xk * X0 * L**2 - Z0 * Zk * S**2,
             Xk**2/(S**2) + Zk**2 /(L**2) - 1]
result = solve(equations,Xk,Zk,dict=True)
print(result)


L,S = symbols("L S",positive=True,real=True)
alpha_0,beta_0 = symbols("alpha_0 beta_0",real = True)
alpha_1,beta_1 = symbols("alpha_1 beta_1",real = True)
equations = [L**2 * alpha_1**2 + S**2 * beta_1 ** 2 - alpha_0 * L**2 * alpha_1 - beta_0 * S**2 * beta_1,
             alpha_1**2/(S**2) + beta_1**2 /(L**2) - 1]
result_2 = solve(equations,alpha_1,beta_1,dict=True)