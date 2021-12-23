import torch
import numpy as np

def noisy_hill(tensor, lib=torch):
    x, y = tensor
    def __fun(x, y, x_mean, y_mean, x_sig, y_sig, lib=lib):
        x, y = tensor
        x_mean, x_sig = tensor
        y_mean, y_sig = tensor
        x_exp, y_exp = tensor
        normalizing = 1 / (2 * np.pi * x_sig * y_sig)
        x_exp = (-1 * lib.square(x - x_mean)) / (2 * lib.square(x_sig))
        y_exp = (-1 * lib.square(y - y_mean)) / (2 * lib.square(y_sig))
        return normalizing * lib.exp(x_exp + y_exp)
    # 3rd local minimum at (-0.5, -0.8)
    z = -1 * __fun(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.35, y_sig=0.35)
    # three steep gaussian trenches
    z -= __fun(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= __fun(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= __fun(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
    return z

def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x ** 2 - A * lib.cos(x * math.pi * 2))
        + (y ** 2 - A * lib.cos(y * math.pi * 2))
    )
    return f

def noisy_hill(tensor, lib=torch):
    x, y = tensor
    return -1 * lib.sin(x * x) * lib.cos(3 * y * y) * lib.exp(-(x * y) * (x * y)) - lib.exp(-(x + y) * (x + y))

def peaks(tensor, lib=torch):
    x, y = tensor
    return 1*(0.6-x)**2*lib.exp(-x**2-(y+1)**2)-2*(x/2.5-(0.5*x)**3-y**5)*lib.exp(-x**2-y**2)-1/3*lib.exp(-(x+1)**2-y**2)

def beals(tensor, lib=torch):
    x, y = tensor
    return (1.5-x+x*y)**2+(2.25-x+x*y**2)**2+(2.625-x+x*y**3)**2

def mishra(tensor, lib=torch):
    x, y = tensor
    return lib.sin(y)*lib.exp((1-lib.cos(x))**2)+lib.cos(x)*lib.exp((1-lib.sin(y))**2)+(x-y)**2

def townsend(tensor, lib=torch):
    x, y = tensor
    return -lib.cos((x-0.1)*y)**2-x*lib.sin(3*x+y)

def goldsteinprice(tensor, lib=torch):
    x, y = tensor
    return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))

def rosenbrock(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def saddle(tensor, lib=torch):
    x, y = tensor
    return x**2-y**2

def saddle2(tensor, lib=torch):
    x, y = tensor
    return x**3-3*x-2*y**2

def cross_sectional(tensor, lib=torch):
    x, y = tensor
    W = [3.5, 0.35, 3.2, -2.0, 1.5, -0.5]
    B = [0.5, 0.50, 0.5,  0.5, 0.1,  0.3]
    def __sigmoid(x, y, w, lib=lib):
        x, y = tensor
        return 1. / (1. + lib.exp(-(x * w + y)))
    err = 0
    for w, b in zip(W, B):
        err += 0.5 * (__sigmoid(x, y, w) - b) ** 2
    return err

def mccormick(tensor, lib=torch):
    x, y = tensor
    return lib.sin(x+y)+(x-y)**2-1.5*x+2.5*y+1

def threehump(tensor, lib=torch):
    x, y = tensor
    return 2*x**2-1.05*x**4+x**6/6+x*y+y**2

def sigmoid(tensor, lib=torch):
    x, y = tensor
    return 1/(lib.exp(-x)+1)

def sigmoid1(tensor, lib=torch):
    x, y = tensor
    return 1/(lib.exp(-(x*y+(0.0001*x)**2+(0.0001*y)**2))+1)

def squre(tensor, lib=torch):
    x, y = tensor
    return (0.1*x)**4+(0.1*y)**4