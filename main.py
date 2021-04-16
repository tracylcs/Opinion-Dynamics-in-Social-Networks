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
s = np.array(np.mat('1;0;0'))
P = np.array(np.mat('0 0.5 0.5;0.5 0 0.5;0.5 0.5 0'))
l = np.array([0.99, 0.01, 0.01])
u = np.array([0.99, 0.99, 0.99])
a0= np.array([0.99, 0.01, 0.01])
b = 1
problem = osp.OpinionSusceptibilityProblem(n, s, P, l, u, a0, b)
opt, path = problem.projected_gradient_algorithm(maxiter=10, maxt=10, gtol=1e-14, print_every=1)
print("L1-budgeted optimal solution:", opt)
rand = problem.random_sampling(size=500)
opt_rand = [problem.a0[i]+problem.b*rand.loc[rand['fx'].idxmin(),'x'+str(i)] for i in range(n)]
print("L1-budgeted optimal solution:", opt_rand)
problem.plot3d(rand, path)

# Example run 3: Randomly generated problem with n=100
np.random.seed(21)
par = rp.param(100)
problem = osp.OpinionSusceptibilityProblem(*par)
opt, path = problem.projected_gradient_algorithm(maxiter=200, maxt=10000, gtol=1e-14, print_every=10)

# Example run 4: Problem from data file (Dataset A)
sym = "A"
n = int(np.loadtxt("./dataset/"+sym+"/n.txt"))
s = np.loadtxt("./dataset/"+sym+"/s.txt", ndmin=2)
P = np.loadtxt("./dataset/"+sym+"/P.txt")
l = np.loadtxt("./dataset/"+sym+"/l.txt")
u = np.loadtxt("./dataset/"+sym+"/u.txt")
a0 = np.loadtxt("./dataset/"+sym+"/a0.txt")
b = int(np.loadtxt("./dataset/"+sym+"/b.txt"))
problem = osp.OpinionSusceptibilityProblem(n,s,P,l,u,a0,b)
opt, path = problem.projected_gradient_algorithm(maxiter=n*2, maxt=10000, gtol=1e-14, print_every=10)
