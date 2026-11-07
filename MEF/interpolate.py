import numpy as np


def lagrange(x0, X, Y):
    n = len(X)
    S = 0
    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                L = L * ((x0 - X[j]) / (X[i] - X[j]))

        S = S + Y[i] * L

    return S
    
