import numpy as np
import matplotlib.pyplot as plt
from dados_ex_04 import x, fx

FLAG_PLOT_DADOS = False
if FLAG_PLOT_DADOS:
    plt.figure(figsize=(8, 5))
    plt.scatter(x, fx)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Diagrama de dispersão')
    plt.grid(True)
    plt.show()
    exit()


# definir mm=m+1 e nn=n+1
mm = len(fx) # numero de pontos discretos
nn = m + 1       # numero de funcoes da base

# definindo as funcoes de base
def phi_P(x, idx):
    if idx == 0:
        return 1
    elif idx == 1:
        return x
    elif idx == 2:
        return x**2
    else:
        raise ValueError("Sem funcao base para este idx")


def phi(t, k, T):
    if k == 0:
        return np.ones_like(t, dtype=float)
    else:
        return np.cos(2.0*np.pi*k*t/T)

## ------------ Resolucao: Forma A ------------
# Construindo N e b do sistema N c = b
N = np.zeros((nn, nn))
b = np.zeros(nn)

for i in range(nn):
    for j in range(nn):
        for k in range(mm):
            N[i, j] += phi(x[k], i) * phi(x[k], j)

for i in range(nn):
    for k in range(mm):
        b[i] += fx[k] * phi(x[k], i)

# resolvendo o sistema N c = b
c = np.linalg.solve(N, b)

print(c)

## ------------ Resolucao: Forma B ------------
# Construindo A e f do sistema A.T A c = A.T f
A = np.zeros((mm, nn))
for i in range(mm):
    for j in range(nn):
        A[i, j] = phi(x[i], j)

N = A.T @ A
b = A.T @ fx

# resolvendo o sistema N c = b
c = np.linalg.solve(N, b)

print(c)

## ------------ Resolucao: Forma C ------------
# Construindo A e f do sistema A.T A c = A.T f
A = np.zeros((mm, nn))
for j in range(nn):
    A[:, j] = phi(x, j)

N = A.T @ A
b = A.T @ fx

# resolvendo o sistema N c = b
c = np.linalg.solve(N, b)

print(c)

## ------------ Visualizacao ------------
# gx(x) = c0 phi_0(x) + c1 phi_1(x)  + c2 phi_2(x)  + ...
def gx(x, coef):
    gx_vec = np.zeros_like(x, dtype=float)
    for i in range(len(gx_vec)):
        for j in range(nn):
            gx_vec[i] += coef[j] * phi(x[i], j, T)
    return gx_vec

FLAG_PLOT = True
if FLAG_PLOT:
    x_plot = np.linspace(np.min(x), np.max(x), 300)
    gx_plot = gx(x_plot, c)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, fx, s=60, label="Dados experimentais")
    plt.plot(x_plot, gx_plot, linewidth=1, color='red', label="Ajuste")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Ajuste por mínimos quadrados')
    plt.grid(True)
    plt.legend()
    plt.show()
