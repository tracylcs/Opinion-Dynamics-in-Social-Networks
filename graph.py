import numpy as np

def DFS(G, i, visited):
    visited[i] = True
    for j in range(G.shape[1]):
        if (G[i,j] == 1 and visited[j] == False):
            return DFS(G, j, visited)

def reverseGraph(G):
    return np.transpose(G)
  
def isSC(G):
    # Check strong connectivity of an adjacency matrix
    n = G.shape[0]
    visited = [False]*n
    DFS(G, 0, visited)
    if any(i == False for i in visited): 
        return False
    GR = reverseGraph(G)
    visited = [False]*n
    DFS(GR, 0,visited)
    if any(i == False for i in visited): 
        return False
    return True