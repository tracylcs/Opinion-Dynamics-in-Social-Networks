import numpy as np
import graph

def get_P(n):
    while (True):
        P = np.round(np.random.rand(n, n),0)
        if (graph.isSC(P)): # to ensure the underlying graph of P is strongly connected
            break
    P = P*np.random.rand(n, n)
    for i in range(n):
        P[i,:] = P[i,:]/np.sum(P[i,:])
    return P

def param(n):
    s = np.random.rand(n,1)
    P = get_P(n)
    l = np.random.rand(n)
    u = np.random.rand(n)
    while (any((l[i]==0 or u[i]==0) for i in range(n))) or (any(l[i]==u[i] for i in range(n))):
        l = np.random.rand(n)
        u = np.random.rand(n)
    for i in range(n):
        if (l[i] > u[i]):
            tmp = l[i]
            l[i] = u[i]
            u[i] = tmp
    a0 = np.random.uniform(l, u)
    b = np.random.uniform(0, n/6)
    return n, s, P, l, u, a0, b