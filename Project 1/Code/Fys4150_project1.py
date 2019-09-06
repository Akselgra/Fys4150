import numpy as np
import matplotlib.pyplot as plt


class Second_derivative():
    
    def __init__(self, v):
        self.n = len(v) #length of array  we want to derivate
        
        
    def builder(self):
        """
        Builds a matrix A
        Used in the linear transformation
        Av = f
        to find the second derivatives of v
        """
        init_array = np.zeros(self.n)
        init_array[0] = 2; init_array[1] = -1; init_array[-1] = -1
        
        init_array_0 = np.zeros(self.n)
        init_array_0[0] = 2; init_array_0[1] = -1;
            
        init_array_1 = np.zeros(self.n)
        init_array_1[-2] = -1; init_array_1[-1] = 2
        
        A = []
        A.append(init_array_0)
        for i in range(1, self.n-1):
            A.append(np.roll(init_array, i))
        A.append(init_array_1)
        A = np.array(A)
        self.A = A
        


if __name__ == "__main__":
    x = np.linspace(0, 8*np.pi, 5000)
    v = np.cos(x)
    dx = x[1]-x[0]
    objekt = Second_derivative(v)
    objekt.builder()
    out = objekt.A
    func = np.dot(out, v)*dx**-2
    plt.plot(x[1:-1], func[1:-1])