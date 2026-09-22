#a) Write a program to calculate E(x) for values of x from 0 to 3 in steps of 0.1. 
# Choose for yourself what method you will use for performing the integral and a suitable number of slices.
#b) When you are convinced your program is working, extend it further to make a graph of E(x) as a function of x.

import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxw
import argparse as ap 

parser = ap.ArgumentParser(description='A tool to calculate the value of a function of the integral of e raised to negative t squared from 0 (or another value) to X')
parser.add_argument('--start', type=int, default = 0, help='Provide the lower limit of integration. This is where the function E, integrating f(t), begins. The start defaults to zero. Make sure you include the two dashes before the variable when passing a value to start.')
parser.add_argument('--X', type=float, default = 3, help='Provide the upper limit of integration. This is where the function E, integrating f(t), ends. X defaults to 3. Make sure you include the two dashes before the variable when passing a value to X.')
parser.add_argument('--deltax', type=float, default = 0.1, help='Provide the size of the step for calculation of the integral. deltax defaults to 0.1. Make sure you include the two dashes before the variable when passing a value to deltax.')

args = parser.parse_args()

start = args.start
X = args.X
deltax = args.deltax

def f(t): #this is the function being integrated withn the function that we are calculating
    return np.exp(-((t)**2)) 

#I. using Newton - Cotes methods
#A. Trapezoid Rule
def trapezoidal_integrating(a,X,deltax): 
    t_space = np.arange(a,X,deltax)
    E = deltax * (0.5*f(t_space[0]) + 0.5*f(t_space[-1]) + np.sum(f(t_space[1:-1]))) #the array is the set of values we are using to calculate the integral; instead of using k times delta x in a for loop, we cycle through the inner values of the array
    return E

trapezoidal_integration = trapezoidal_integrating(start,X,deltax) 

#B. Simpsons Rule
def simpson_integrating(a,X,deltax): 
    t_space = np.arange(a,X,0.1)
    E = (deltax/3) * (f(t_space[0]) + f(t_space[-1])+ 4*np.sum(f(t_space[1:-1:2]))+2*np.sum(f(t_space[2:-1:2])))
    return E

simpson_integration = simpson_integrating(start,X,deltax)

#Error
def E_M_error_trapezoidal(a,X):
    error_t_space = np.arange(a,X,0.2)
    trapezoidal_e2 = (1/3)*(simpson_integration-simpson_integrating(start,X,0.2))
    return trapezoidal_e2

def E_M_error_simpson(a,X):
    error_t_space = np.arange(a,X,0.2)
    simpson_e2 = (1/15)*(simpson_integration-simpson_integrating(start,X,0.2))
    return simpson_e2


#II. Romberg 
def romberg_integrating(a, X, max_level, h_start):
    R = np.zeros((max_level, max_level))
    for i in range(max_level):
        h_i = h_start / (2**i)  # halves step size each row
        R[i, 0] = trapezoidal_integrating(a, X, h_i)
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    return R[max_level-1, max_level-1]

romberg_integration = romberg_integrating(start, X, 4, deltax)

n=int((X-start)//deltax)
#Gaussian Quadrature
def gaussian_integrating(n):
    h,w = gaussxw(n)
    hp = 0.5*(X-start)*h + 0.5*(X+start)
    wp = 0.5*(X-start)*w

    # Perform the integration
    s = 0.0
    for k in range(n):
        s += wp[k]*f(hp[k])

    return s
gaussian_integration = gaussian_integrating(n)


#percentage difference from the expected answer
expected_answer = 0.5*np.sqrt(np.pi) #result of the gaussian integral substituting in r and solving for square root of pi over 2
trapezoidal_difference = (trapezoidal_integration-expected_answer)/expected_answer
simpson_difference = (simpson_integration-expected_answer)/expected_answer
romberg_difference = (romberg_integration-expected_answer)/expected_answer
gaussian_difference = (gaussian_integration-expected_answer)/expected_answer

if __name__ == '__main__':
    print(f"NEWTON-COTES METHODS (all calculations with step size of {deltax})")
    print(f"    TRAPEZOIDAL: The result of the integral calculated through the trapezoidal method is {trapezoidal_integration}")
    print(f"    EULER-MACLAURIN ERROR FOR THE TRAPEZOIDAL METHOD: the error in the trapezoidal method calculation is {E_M_error_trapezoidal(start,X)} with steps of {deltax} \n")
    print(f"\n  SIMPSON'S: The result of the integral calculated through the Simpson's rule method is {simpson_integration}")
    print(f"    EULER-MACLAURIN ERROR FOR THE SIMPSON'S METHOD: the error in the Simpson's rule method calculation is {E_M_error_simpson(start,X)} with steps of {deltax} \n")
    print("\nROMBERG METHOD")
    print(f"    ROMBERG: The result of the integral calculated through the Romberg method is {romberg_integration}\n")
    print("\nGAUSSIAN QUADRATURE")
    print(f"    The result of the integral calculated using Gaussian quadrature is {gaussian_integration}\n")
    print("\nDIFFERENCES FROM THE EXPECTED RESULT")
    print(f"    The expected answer is {expected_answer}")
    print(f"    The percentage difference from the expected answer is {trapezoidal_difference} through the trapezoidal method of computing the integral.")
    print(f"    The percentage difference from the expected answer is {simpson_difference} through Simpson's method of computing the integral.")
    print(f"    The percentage difference from the expected answer is {romberg_difference} through the Romberg method of computing the integral.")
    print(f"    The percentage difference from the expected answer is {gaussian_difference} through the gaussian quadrature computation of the integral.")

    graph_input = input(str("do you want a graph? yes or no"))
    if graph_input.lower() in ("yes", "y", "yeah", "ye"):
        t_space = np.arange(start, X, deltax)
        y = f(t_space)
        trap_areas = deltax * (y[:-1] + y[1:]) / 2
        E_cumulative = np.concatenate(([0], np.cumsum(trap_areas))) #I switched to a different approach with a cumulative sum because my script was running very slow.
        plt.plot(t_space, y, color='red', label='f(t)')
        plt.plot(t_space, E_cumulative, color='blue', label='E(x)')
        plt.xlabel('t')
        plt.ylabel('E(X) and f(t)')
        plt.legend()
        plt.title("E(x); a function of the integral of f(t)")
        plt.show()
    else:
        print("alrighty!")