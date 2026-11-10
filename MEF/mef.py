import numpy as np
from interpolate import*
import matplotlib.pyplot as plt

a = 0
b = 1
Ne = 9
def mesh(a, b, N_element):
    dim = 2
    omega = np.linspace(a, b, Ne + 1, endpoint = True)
    nodes = []
    elements = np.zeros((N_element, dim))
    for node in omega:
        nodes.append(node)

    for e in range(N_element):
        elements[e, 0] = nodes[e]
        elements[e, 1] = nodes[e + 1]

    return np.array(nodes).reshape(-1,1), np.array(elements)


