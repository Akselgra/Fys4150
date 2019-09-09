import numpy as np
import matplotlib.pyplot as plt
from time import time


class Project_solver():
    
    def __init__(self, f, a = -1, b = 2, c = -1, M = False):
        self.n = len(f) 
        self.h = 1/(self.n + 1)
        self.f = f
        self.a = a
        self.b = b
        self.c = c
        if M == True:
            self.M = M
        else:
            self.builder()
    def builder(self):
        """
        Builds a matrix M
        Used in the linear transformation
        Mv = f
        """
        M = np.zeros((self.n, 4))
        M[0, :] = np.array([self.b, self.c, 0, self.f[0]])
        for i in range(1, self.n-1):
            M[i, :] = np.array([self.a, self.b, self.c, self.f[i]])
        M[-1, :] = np.array([0, self.a, self.b, self.f[-1]])
        self.M = M
        self.M[:, :-1] = self.M[:, :-1]/self.h**2


    def linear_equation_solver(self):
        """
        Solves Mv = f for v
        """
        self.M[0, :] = self.M[0, :]/self.M[0, 0]
        self.M[1, :] = self.M[1, :] - self.M[0, :]*self.M[1, 0]
        for i in range(1, self.n-2):
            self.M[i, 1:] = self.M[i, 1:]/self.M[i, 1]
            
            multi_comp = self.M[i+1, 0]
            for j in range(2):
                self.M[i+1, j] = self.M[i+1, j] - self.M[i, j+1]*multi_comp
            self.M[i+1, -1] = self.M[i+1, -1] - self.M[i, -1]*multi_comp

        self.M[-2, :] = self.M[-2, :]/self.M[-2, 1]
        self.M[-1, :] = self.M[-1, :] - self.M[-2, :]*self.M[-1, 1]
        self.M[-1, :] = self.M[-1, :]/self.M[-1, -2]
                
        
        self.M[-2, :] = self.M[-2, :] - self.M[-1, :]*self.M[-2, -2]
        
        self.M[0, -1] = 0
        for i in range(1, self.n-1):
            multi_comp = self.M[-(i+1), -2]
            self.M[-(i+1), -2] = \
            self.M[-(i+1), -2] - self.M[-i, -3]*multi_comp
            self.M[-(i+1), -1] = self.M[-(i+1), -1] - self.M[-i, -1]*multi_comp
            
        self.M[0, :] = self.M[0, :] - self.M[1, :]*self.M[0, 1]
        self.v = self.M[:, -1]
        self.M[-1, -1] = 0
        self.M[0, -1] = 0
        self.v = self.M[:, -1]
        
        
if __name__ == "__main__":
    def trueplot(x):
        return(1 - (1 - np.exp(-10))*x - np.exp(-10*x))
    
    def linear_equation_tester():
        n_list = [int(1e1), int(1e2), int(1e3)]
        for n in n_list:
            x = np.linspace(0, 1, n)
            f = 100*np.exp(-10*x)
            objekt = Project_solver(f)
            objekt.linear_equation_solver()
            plt.plot(x, objekt.v)
            plt.plot(x, trueplot(x))
            plt.legend(["numerical", "closed-form solution"])
            plt.title("Numerical vs closed-form for n = %g" % n)
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.show()
        
    linear_equation_tester()