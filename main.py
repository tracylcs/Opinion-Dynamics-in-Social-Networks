import numpy as np
import opinion_susceptibility_problem as osp
import rand_param as rp

# Example run 1: Self-defined problem with n=2
n = 2
s = np.array(np.mat('1;0'))
P = np.array(np.mat('0 1;1 0'))
l = np.array([0.2, 0.2])
u = np.array([0.8, 0.8])
a0= np.array([0.5, 0.5])
b = 0.4
problem = osp.OpinionSusceptibilityProblem(n, s, P, l, u, a0, b)
opt, path = problem.projected_gradient_algorithm(maxiter=10, maxt=10, gtol=1e-14, print_every=1)
print("L1-budgeted optimal solution:", opt)
rand = problem.random_sampling(size=50)
opt_rand = [problem.a0[i]+problem.b*rand.loc[rand['fx'].idxmin(),'x'+str(i)] for i in range(n)]
print("L1-budgeted optimal solution:", opt_rand)
problem.plot2d(rand, path)

# Example run 2: Self-defined problem with n=3
n = 3
s = np.array(np.mat('1;1;0'))
P = np.array(np.mat('0 0.5 0.5;0.5 0 0.5;0.5 0.5 0'))
l = np.array([0.2, 0.2, 0.2])
u = np.array([0.8, 0.8, 0.8])
a0= np.array([0.5, 0.5, 0.5])
b = 0.5
problem = osp.OpinionSusceptibilityProblem(n, s, P, l, u, a0, b)
opt, path = problem.projected_gradient_algorithm(maxiter=10, maxt=10, gtol=1e-14, print_every=1)
print("L1-budgeted optimal solution:", opt)
rand = problem.random_sampling(size=500)
opt_rand = [problem.a0[i]+problem.b*rand.loc[rand['fx'].idxmin(),'x'+str(i)] for i in range(n)]
print("L1-budgeted optimal solution:", opt_rand)
problem.plot3d(rand, path)

# Example run 3: Randomly generated problem with n=100
np.random.seed(4801)
par = rp.param(100)
problem = osp.OpinionSusceptibilityProblem(*par)
opt, path = problem.projected_gradient_algorithm(maxiter=120, maxt=200, gtol=1e-14, print_every=10)