import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import  linalg as LA
def spectral_clustering(adj_matrix):
    D=np.zeros(adj_matrix.shape,dtype=float)
    for i in range(D.shape[0]):
        D[i][i]=np.sum(adj_matrix[i])
    Laplacian=np.subtract(D,adj_matrix)
    eigen_values,eigen_vectors=LA.eig(Laplacian)
    second_eigen_vector=eigen_vectors[:,1]
    tem_condtion=second_eigen_vector<0
    partition=tem_condtion.astype(int)
    return partition


color_map=[]
i=0
g=nx.Graph()
g.add_nodes_from(range(1,9))
g.add_edges_from([(1,2,{'weight':5}),(1,3,{'weight':8}),(1,4,{'weight':7}),(2,3,{'weight':10}),(2,4,{'weight':12}),(3,4,{'weight':6}),(3,5,{'weight':2}),(5,6,{'weight':9}),(6,7,{'weight':10}),(5,7,{'weight':11}),(3,8,{'weight':6}),(5,8,{'weight':6})])
A=nx.to_numpy_matrix(g)
partition=spectral_clustering(A)
for node in g:
    if partition[i]==0:
        color_map.append('red')
    else:
        color_map.append('blue')
    i+=1

plt.subplot(121)
nx.draw(g,node_color=color_map,with_labels=True, font_weight='bold')
plt.show()

