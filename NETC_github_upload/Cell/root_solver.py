import numpy
from scipy.optimize import fsolve
import numpy as np
import networkx as nx

np.random.seed(369)
dim = 100
neighbor = 4
G = nx.watts_strogatz_graph(dim, neighbor, 0.5)
A = nx.to_numpy_matrix(G)
A = numpy.array(A)
np.save('./data/A_100',A)
def Cell(X):
    B = 1
    h = 2
    # scale = 10
    Y = X.reshape(dim,1)
    return -B*X + np.matmul(A,Y**2/(1+Y**2)).T[0]

for i in range(200):
    X0 = np.random.uniform(0,10,[dim])
    result = fsolve(Cell, X0)
    print(result)
np.save('./data/target_100',np.array(result))
'''
[5.72887493 2.90587157 3.63921432 3.71516486 2.82402557 3.50299645
 2.68868918 1.80106209 3.57665467 1.88065846 1.87536705 4.39905647
 5.5992888  1.90716    2.78441521 1.8978953  2.74865214 5.40766408
 3.7140467  3.59752592 3.76836057 3.73948297 3.82556171 3.70843021
 4.62104559 3.76645921 2.76868405 2.74708566 3.6569334  3.70385351
 4.73378468 3.49936553 4.72953546 1.89670138 3.60879603 5.50095148
 5.69630664 4.54679397 1.8270845  2.82037907 2.62448807 3.7426017
 2.89358267 4.68760135 4.62869059 3.83885443 3.67213359 2.62854187
 3.7104445  3.71083356 2.8118029  1.89127954 3.48930052 2.79255285
 2.74343985 3.67199214 5.60481894 3.75523268 3.76491059 2.81851017
 4.66329793 1.85292291 3.65884215 2.81046859 3.74430588 4.50030855
 4.66598192 3.81047285 3.73627566 4.77179404 4.77369262 4.70975318
 2.7591438  2.6980942  2.83870972 3.68708231 3.77636901 3.81556536
 7.49921504 3.80264621 4.62147311 3.69908556 3.82083663 3.80370804
 4.56509652 4.51986456 4.67029719 1.90722832 3.7894504  5.42852359
 1.80952597 3.67458377 2.84910816 5.60431736 4.6316767  4.8157287
 3.79492573 5.59892605 4.54851601 3.79354728]
'''
