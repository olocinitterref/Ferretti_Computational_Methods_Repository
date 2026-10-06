import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import astropy.constants as const
import astropy.units as u

parser = ap.ArgumentParser(description='Find the Earth-Moon L1 Lagrange point')
parser.add_argument('--guess_1', type=float, default=1.844e8, help='Lower limit of the guess range (m).')
args = parser.parse_args()
guess1 = args.guess_1 * u.m
#I set up this variable because I was using set floats when I first coded this, and I didn't want to type args over and over

M_moon = 7.348e22 * u.kg
R = 3.844e8 * u.m
w = 2.662e-6 / u.s            # angular velocity
G = const.G
M_earth = const.M_earth

def f(r):
    return (w**2*r**5 - 2*w**2*r**4*R + w**2*r**3*R**2
            - G*M_earth*r**2 + G*M_moon*r**2
            - G*M_earth*R**2 + 2*G*M_earth*R*r)

def dfdr(r):
    return (5*w**2*r**4 - 8*w**2*r**3*R + 3*w**2*r**2*R**2
            - 2*G*M_earth*r + 2*G*M_moon*r + 2*G*M_earth*R)

def newton_method(r0, tol=1e-6*u.m, max_iter=10000): #the loop in the function stops when the step falls below 10-6, and does not run more than 10,000 iterations.
    r = r0
    for _ in range(max_iter): #the actual netwon's metho calculation
        slope = dfdr(r)
        if slope.value == 0:
            return None
        step = f(r) / slope          # units: meters
        r = r - step
        if abs(step) < tol:
            return r
    return None

roots = [] #roots will collect in this array

if __name__ == '__main__':
    for r0 in np.linspace(0, R, 50):      # Quantity array in meters
        root = newton_method(r0) #this loop allows the script to find every root within a range using the newton_method function 
        if root is not None and 0*u.m <= root <= R and not any(abs(root - s) < 1*u.m for s in roots): #this checks that the iterations of Newton's method converge, that there are no duplicates, and that roots are between the earth and the moon.
            roots.append(root)

    print(f"all roots in [{0}, {R}]:")
    for root in sorted(roots):
        print(f"The first Lagrange point between the earth and the moon is located at r = {root.to_value(u.m):.4e} m  (r/R = {(root/R).to_value(u.dimensionless_unscaled):.4f})")
    r_L1 = roots[0].to_value(u.m) #need to convert quantities to values

    # Plot
    r_space = np.linspace(0.01, 0.99, 1000) * R
    plt.plot(r_space.to_value(u.m), f(r_space).to_value(u.m**5/u.s**2), color='red', label='f(r)')
    plt.axhline(y=0, color='black')
    plt.scatter(r_L1, 0, color='black', zorder=5, label='L1')
    plt.annotate(f"Lagrange Point\nat {r_L1:.4e} m",
                 xy=(r_L1, 0),
                 xytext=(-35, -25), textcoords='offset points')
    plt.legend(); plt.xlabel('r (m)'); plt.ylabel('f(r) (m$^5$/s$^2$)')
    plt.title("Locating the first Lagrange Point")
    plt.show()