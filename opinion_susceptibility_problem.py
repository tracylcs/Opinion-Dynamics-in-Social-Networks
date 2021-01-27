import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import time
import pandas as pd

class OpinionSusceptibilityProblem:
    
    def __init__(self, n, s, P, l, u, a0, b):
        
        self.n = n   # int
        self.s = s   # numpy array (n,1)
        self.P = P   # numpy array (n,n)
        self.l = l   # numpy array (n,)
        self.u = u   # numpy array (n,)
        self.a0 = a0  # numpy array (n,)
        self.b = b   # float
        
        # The following attributes are evaluated based on the inputs
        self.x_lb = (self.l-self.a0)/self.b
        self.x_lb = np.where(self.x_lb<-1, -1, self.x_lb)
        self.x_ub = (self.u-self.a0)/self.b
        self.x_ub = np.where(self.x_ub>1, 1, self.x_ub)
    
    def optimistic_local_search(self, maxiter=200):   
        # Is it necessary to set the 'maxiter' here?
    
        # Implementation of the Optimistic Local Search (Algorithm 2 in Chan et al., 2019 [4])
        # Return an unbudgeted optimal solution
        
        n = self.n
        s = self.s + np.random.rand(n,1)*1e-8
        a = self.u.copy()
        eps_a = np.amin(a)
        z = np.ones((n,1))
        t = 0
        err = np.power(1-eps_a, t)/eps_a
        I = np.identity(n)
        
        count = 0
        while (any(np.abs(s[i]-z[i]) <= err for i in range(n)) and count < maxiter):
            z = np.matmul(np.diag(a),s) + np.matmul(I-np.diag(a), np.matmul(self.P,z))
            t = t+1
            for i in range(n):
                if (z[i] <= s[i] and a[i] == self.u[i]):
                    a[i] = self.l[i]
                    t = 0
                elif (z[i] > s[i] and a[i] == self.l[i]):
                    a[i] = self.u[i]
                    t = 0
            eps_a = np.amin(a)
            err = np.power(1-eps_a, t)/eps_a
            count += 1
        
        print("Unbudgeted optimal value:", self.f_approx((a-self.a0)/self.b))
        return a
           
    def random_sampling(self, size):
        
        # Return a dataframe containing the evaluation of the objective function
        # at random points in the search space of the L1-budgeted variant
        
        n = self.n
        feval = pd.DataFrame([], columns = ['x'+str(i) for i in range(n)]+['fx'])
        minfx = 1
                
        # Evaluate the objective function at the extreme points of the 
        # unbudgeted variant that lies in the feasible region of the 
        # L1-budgeted variant
        for a in product(*([self.l[i],self.u[i]] for i in range(n))):
            x = (a-self.a0)/self.b
            if (np.sum(np.abs(x))<=1):
                fx = self.f_approx(x)
                feval = feval.append(pd.Series(list(x)+[fx], 
                                               index=['x'+str(i) for i in range(n)]+['fx']), 
                                     ignore_index=True)
                if (fx < minfx):
                    minfx = fx
        
        # Evaluate the objective function at random points on the 
        # boundary of the 1-norm ball that lie in the feasible region 
        # of the L1-budgeted variant
        for direc in product(*([-1,1] for i in range(n))):
            success = 0
            trial = 0
            while (success < size and trial < 2*size):
                distr = np.random.dirichlet(np.ones(n))
                x = np.multiply(direc, distr)
                if all(self.x_lb[i] <= x[i] and x[i] <= self.x_ub[i] for i in range(n)):
                    fx = self.f_approx(x)
                    feval = feval.append(pd.Series(list(x)+[fx], 
                                                   index=['x'+str(i) for i in range(n)]+['fx']), 
                                         ignore_index=True)
                    if (fx < minfx):
                        minfx = fx
                    success += 1
                trial += 1
                        
        print("L1-budgeted optimal from Random Sampling:", minfx)
        
        return feval

    def plot2d(self, rand=None, path=None):
        
        # Produce a matplotlib plot when n=2
        # rand: a dataframe from random_sampling
        # path: a dataframe from projected_gradient_algorithm
        
        if (rand is not None and not rand.empty):
            im = plt.scatter(rand.x0, rand.x1, c=rand.fx, alpha=0.8, s=5, cmap='viridis_r')
            rand_best = rand.loc[rand['fx'].idxmin()]
            plt.scatter(rand_best.x0, rand_best.x1, c='r', s=50)
            plt.colorbar(im)
            
        if (path is not None and not path.empty):
            plt.plot(path.x0, path.x1, '-x', color='black', ms=10)
            plt.scatter(path.head(1).x0, path.head(1).x1, color='black', s=50) 
            
        return
    
    def plot3d(self, rand=None, path=None):
        
        # Produce a matplotlib plot when n=2
        # rand: a dataframe from random_sampling
        # path: a dataframe from projected_gradient_algorithm
        
        fig = plt.figure(figsize=(40,40))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        if (rand is not None and not rand.empty):
            im = ax.scatter3D(rand.x0, rand.x1, rand.x2, c=rand.fx, s=5, cmap='viridis_r')
            rand_best = rand.loc[rand['fx'].idxmin()]
            ax.scatter3D(rand_best.x0, rand_best.x1, rand_best.x2, c='r', s=100)
            fig.colorbar(im)
            
        if (path is not None):
            ax.plot3D(path.x0, path.x1, path.x2, '-x', c='black', ms=50)
            ax.scatter3D(path.head(1).x0, path.head(1).x1, path.head(1).x2, c='black', s=50)
            
        return

    def f_exact(self, a):
        
        # Take a resistance parameter alpha as input
        # Return the exact objective function value
        
        I = np.identity(self.n)
        M = inv(I - np.matmul((I-np.diag(a)), self.P))
        z = np.matmul(M, np.matmul(np.diag(a), self.s))
        return np.average(z)

    def df_dx_exact(self, x):
        
        # Take a resistance parameter alpha as input
        # Return the exact gradient vector
        
        a = self.a0 + self.b * x
        I = np.identity(self.n)
        M = inv(I - np.matmul((I-np.diag(a)), self.P))
        z = np.matmul(M, np.matmul(np.diag(a), self.s))
        return self.b/self.n*np.multiply(np.multiply(M.sum(axis=0)[:,None], np.reshape(1/(1-a), (self.n,1))), self.s-z)
  
    def z_approx(self, x, eps=1e-4):
         
        # Take a budget distribution parameter x as input
        # Return an approximated equilibrium opinion vector
        # Reference: Chan et al., 2019 [4]
               
        a = self.a0 + self.b*x
        n = self.n
        eps_a = np.amin(a)
        it = np.int(np.log(eps*eps_a)/np.log(1-eps_a))+1
        z = np.ones((n,1))
        I = np.identity(n)
        
        for t in range(it):
            z = np.matmul(np.diag(a),self.s)+np.matmul(I-np.diag(a), np.matmul(self.P,z))
        return z
    
    def f_approx(self, x):
        z = self.z_approx(x)
        return np.average(z)
    
    def df_dx_approx(self, x, eps=1e-1):
        
        # Take a budget distribution parameter x as input
        # Return an approximated gradient vector
        # Reference: Abebe et al., 2020 [5]
        
        a = self.a0 + self.b*x
        n = self.n
        eps_a = np.amin(a)
        it = np.int(np.log(eps*eps_a)/np.log(1-eps_a))
        z = np.ones((n,1))
        r = np.ones((n,1))
        I = np.identity(n)
        
        for t in range(it):
            z = np.matmul(np.diag(a),self.s) + np.matmul(I-np.diag(a), np.matmul(self.P,z))
            r = np.ones((n,1)) + np.matmul(np.transpose(self.P), np.matmul(I-np.diag(a),r))
        return self.b/self.n*np.matmul(np.diag(np.reshape(self.s-z, (n,)))*np.diag(1/(1-a)),r)
    
    def design_matrix(self, x, colnum):
        # Return a design matrix built on the direction vector x provided
        C = np.zeros((self.n, colnum))
        col = 0
        for i in range(self.n):
            if (x[i] != 0):
                if (col == colnum):
                    for j in range(colnum):
                        C[i,j] = -x[i]
                    return C
                else:
                    C[i,col] = x[i]
                    col += 1
                    
    def projection_matrix(self, x, colnum):
        # Return the projection matrix based on the direction vector x provided
        if (colnum != 0):
            C = self.design_matrix(x, colnum)
        else:
            C = np.reshape(x,(self.n,1))
        return np.matmul(np.matmul(C,inv(np.matmul(C.T,C))),C.T)
   
    def stepping(self, currentx, direc, d, currentfx):
        
        # Perform the stepping step on a face
        # Return the next point and the corresponding objective value
        
        n = self.n
        u = np.zeros((n,))
        for i in range(n):
            if (direc[i]==-1 and d[i]>0):
                u[i] = self.x_lb[i]
            elif (direc[i]==1 and d[i]<0):
                u[i] = self.x_ub[i]
            else:
                u[i]=0
        
        box = u-currentx
        
        for i in range(n):
            if (box[i]==0):
                box[i]=1
        max_ratio = np.max(np.abs(np.divide(d, box)))
        step = d/max_ratio  # to ensure the next point is inside the feasible region
        
        ftmp = self.f_approx(currentx-step)
        while (ftmp > currentfx): # to ensure the objective value of the next point
                                  # is smaller than that of the current point
            step = step/2
            ftmp = self.f_approx(currentx-step)
        
        x = currentx-step
        fx = ftmp
        
        return x, fx
    
    def face(self, x, g):
        # Return the direction vector direc that indicates on which face
        # the gradient should project
        n = self.n
        direc = np.sign(x)
        for i in range(n):
            if (direc[i]==0):
                if (g[i]>=0):
                    direc[i]=-1
                else:
                    direc[i]=1
        return direc
            
    def projected_gradient(self, x, g):
        
        # Return
        # (1) either the projected gradient vector d, or the index of the
        # gradient component to be extracted
        # (2) the direction vector direc, which indicates on which face the 
        # gradient is projected
        # (3) the number of zeros in direc
        
        n = self.n
        direc = self.face(x, g)
        H = self.projection_matrix(direc, n-1)
        d = np.reshape(np.matmul(H, g),(n,))
        count_zero = 0
        amended = True
        while (amended):
            amended = False
            for i in range(n):
                # check whether the current projection gradient leads to
                # a point outside the feasible region
                if (direc[i]!=0 and ((x[i]==self.x_lb[i] and d[i]>0) or 
                                     (x[i]==self.x_ub[i] and d[i]<0) or 
                                     (x[i]==0 and direc[i]==-1 and d[i]<0) or 
                                     (x[i]==0 and direc[i]==1 and d[i]>0))):
                    direc[i] = 0
                    count_zero += 1
                    amended = True
                    break
            if (count_zero == n-1):
                d = 0
                for i in range(n):
                    if (direc[i]!=0):
                        d = i
                        break
                return d, direc, count_zero
            
            elif (amended):
                H = self.projection_matrix(direc, n-1-count_zero)
                d = np.reshape(np.matmul(H, g),(n,))
                
        return d, direc, count_zero

    def print_termination_details(self, fx, num_hit, time):
        print("Optimal value:", fx)
        print("|{xi|xi=0 or xi=li or xi=ui}|:", num_hit)
        print("Time taken:", round(time, 2), "s")
        return
    
    def projected_gradient_algorithm(self, maxiter, maxt, gtol, print_every):
        
        # The main function that solves the L1-budgeted variant
        # Return the optimal alpha and the dataframe about the search path
        
        print("Projected Gradient Algorithm starts running.")
        start = time.time()
        n = self.n
        
        # Run the Optimistic Local Search to find an unbudgeted optimal
        # Return the unbudgeted optimal if the budget available is sufficient
        # to reach this point
        au = self.optimistic_local_search()
        if (np.sum(np.abs(au-self.a0)) <= self.b):
            print("The budget is adequate to reach the unbudgeted optimal. Projected Gradient Algorithm terminates.")
            return au, None

        path = pd.DataFrame([], columns = ['x'+str(i) for i in range(n)]+
                            ['fx',"|{xi|xi=0 or xi=li or xi=ui}|"])
        
        # Find the initial point of the algorithm
        v = np.where(au < self.a0, self.x_lb, self.x_ub)
        x = v / np.sum(np.abs(v))
        fx = self.f_approx(x)
        num_hit = np.sum(x==0)+np.sum(x==self.x_lb)+np.sum(x==self.x_ub)
        path = path.append(pd.Series(list(x) + [fx, num_hit], 
                                     index=['x'+str(i) for i in range(n)]+
                                     ['fx',"|{xi|xi=0 or xi=li or xi=ui}|"]),
                           ignore_index=True)
        print("Initial:", fx, ", |{xi|xi=0 or xi=li or xi=ui}|:", num_hit)
        
        for t in range(maxiter):
            
            g = self.df_dx_approx(x) # gradient evaluation
            
            if (max(np.abs(g))<gtol): # inf norm of the gradient < gtol
                end = time.time()
                print("The gradient is too small. Projected Gradient Algorithm terminates.")
                self.print_termination_details(fx, num_hit, end-start)
                return self.a0 + self.b*x, path
                        
            d, direc, count_zero = self.projected_gradient(x, g)
                
            if (count_zero == n-1): # the gradient is projected on an edge (1-face)
            
                if (x[d]==0 or (x[d]<0 and g[d]>=0) or (x[d]>0 and g[d]<=0)): 
                        # projected gradient points outside the feasible region
                    end = time.time()
                    print("Projected Gradient Algorithm terminates. A vertex of the feasible region is returned.")
                    self.print_termination_details(fx, num_hit, end-start)
                    return self.a0 + self.b * x, path
                
                else: # projected gradient points into the feasible region
                    if (x[d]<0):
                        x[d]=np.min(-x[d],self.x_ub[d])
                    else:
                        x[d]=np.min(-x[d],self.x_lb[d])
                    direc[d] = np.sign(x[d])
                    fx = self.f_approx(x)
            
            elif (max(np.abs(d)) < gtol): # the gradient is projected on a k-face, where 1<k<n,
                                          # but the inf norm of the projected gradient < gtol
                end = time.time()
                print("The projected gradient is too small. Projected Gradient Algorithm terminates.")
                self.print_termination_details(fx, num_hit, end-start)
                return self.a0 + self.b * x, path
            
            else: # the gradient is projected on a k-face, where 1<k<n,
            
                x, fx = self.stepping(x, direc, d, fx) # perform stepping
            
            tic = time.time()

            num_hit = np.sum(x==0)+np.sum(x==self.x_lb)+np.sum(x==self.x_ub)
            path = path.append(pd.Series(list(x) + [fx, num_hit], 
                                         index=['x'+str(i) for i in range(n)]+['fx', "|{xi|xi=0 or xi=li or xi=ui}|"]), 
                               ignore_index=True)

            if ((t+1)%print_every==0):
                print("Iteration",t+1,":", fx, ", |{xi|xi=0 or xi=li or xi=ui}|:", num_hit)

            if (tic-start>maxt): # running time exceeds maximum
                end = time.time()
                print("Searching time reaches maximum. Projected Gradient Algorithm terminates.")
                self.print_termination_details(fx, num_hit, end-start)
                return self.a0 + self.b * x, path
            
        end = time.time() # number of iterations exceeds maximum
        print("Searching iteration reaches maximum. Projected Gradient Algorithm terminates.")
        self.print_termination_details(fx, num_hit, end-start)
        return self.a0 + self.b * x, path
