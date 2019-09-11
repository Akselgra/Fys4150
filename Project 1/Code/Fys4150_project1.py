import numpy as np
import matplotlib.pyplot as plt
from time import time


class Project_solver():
    
    def __init__(self, f, a = -1, b = 2, c = -1, M = True):
        self.n = len(f) 
        self.h = 1/(self.n + 1)
        self.f = f*self.h**2
        self.a = a
        self.b = b
        self.c = c
        if M == True:
            self.builder()
        else:
            self.M = M
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
        self.M[:, :-1] = self.M[:, :-1]


    def linear_equation_solver(self):
        """
        Solves Mv = f for v
        using forwards and backwards substitution
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
        self.thingyjing = self.M[:, -1]
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
        
        
    def slow_specific_linear_equation_solver(self):
        """
        Solves Mv = f for v
        using the fact that the diagonal elements in M are all equal.
        More generalized.
        """
        
        self.M[0, :] = self.M[0, :]/self.M[0, 0]
        self.M[1, :] = self.M[1, :] - self.M[0, :]*self.M[1, 0]
        
        for i in range(1, self.n-2):
            self.M[i, 2:] = self.M[i, 2:]/self.M[i, 1]
            self.M[i+1, 1] = self.M[i+1, 1] + self.M[i, 2]
            self.M[i+1, -1] = self.M[i+1, -1] + self.M[i, -1]
            
        self.M[-2, :] = self.M[-2, :]/self.M[-2, 1]    
        self.M[-1, :] = self.M[-1, :] - self.M[-2, :]
        self.M[-1, 2:] = self.M[-1, 2:]/self.M[-1, -2]
        self.M[0, -1] = 0
        self.M[-1, -1] = 0
        for i in range(1, self.n-1):
            self.M[-(i+1), -1] = self.M[-(i+1), -1] - \
            self.M[-i, -1]*self.M[-(i+1), -2]
            
        self.M[-1, -1] = 0
        
        
    def relative_error_finder(self):
        """
        finds the relative error epsilon
        for the specific linear equation solver
        epsilon = log10((v - u)/u)
        """
        i = np.arange(1, self.n+1)
        numbers = (i)/(i+1)
        self.fast_specific_linear_equation_solver(numbers)
        x = np.linspace(0, 1, self.n)
        u = (1 - (1 - np.exp(-10))*x - np.exp(-10*x))
        v = self.v
        
        self.epsilon = np.log10(np.abs((v[1:-1] - u[1:-1])/u[1:-1]))
        
        
    def fast_specific_linear_equation_solver(self, numbers):
        """
        Solves Mv = f for v
        for the matrix
        [[2, -1, ...,          f]
         [-1, 2, -1, ...,      f]
         [..., -1, 2, -1, ..., f]
         [...0, 2, -1,         f]]
        using a list of numbers containing what to do with f.
        """
        
        v = self.f
        v[0] = v[0]*numbers[0]
        for i in range(1, self.n-1):
            v[i] = (v[i] + v[i-1])*numbers[i]
        v[-1] = 0
        for i in range(1, self.n-1):
            v[-(i+1)] = v[-(i+1)] + v[-(i)]*numbers[-(i+1)]
        v[0] = 0
        self.v = v
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
    
    def computation_time_test():
        n = int(1e6)
        i = np.arange(1, n+1)
        numbers = i/(i+1)
        
        x = np.linspace(0, 1, n)
        f = 100*np.exp(-10*x)
        
        objekt = Project_solver(f)
        
        start = time()
        objekt.linear_equation_solver()
        stop1 = time()
        objekt.fast_specific_linear_equation_solver(numbers)
        stop2 = time()
        
        print("The first method took %g seconds" % (stop1 - start))
        print("The second method took %g seconds" % (stop2 - stop1))
        print((stop1 - start)/(stop2 - stop1))




        
    def relative_error_tester():
        ns = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
        error_max = []
        for n in ns:
            x = np.linspace(0, 1, n)
            f = 100*np.exp(-10*x)
            
            objekt = Project_solver(f)
            objekt.relative_error_finder()
            error_max.append(np.max(objekt.epsilon))
        print(error_max)
    
    
    def LU_tester():
        ns = [10, 100, 1000]
        a = -1
        b = 2
        c = -1
        
        for n in ns:
            x = np.linspace(0, 1, n)
            f = 100*np.exp(-10*x)
            
            M = np.zeros((n, n))
            M[0, 0] = b
            M[0, 1] = c
            for i in range(1, n-1):
                M[i, i-1] = a
                M[i, i] = b
                M[i, i+1] = c
            M[-1, -2] = a
            M[-1, -1] = b
            
            h = 1/(n+1)
            start = time()
            v_np = np.linalg.solve(M, f*h**2)
            stop1 = time()
            objekt = Project_solver(f)
            objekt.slow_specific_linear_equation_solver()
            stop2 = time()
            v_mine = objekt.M[:, -1]
            
            plt.plot(x, v_np)
            plt.plot(x, v_mine)
            plt.legend(["numpy", "mine"])
            plt.xlabel("x")
            plt.ylabel("v")
            plt.title("Solutions to the linear equations, n = %g" % n)
            plt.show()
            print(stop1 - start)
            print(stop2 - stop1)
    LU_tester()
    #relative_error_tester()
    #computation_time_test()
    