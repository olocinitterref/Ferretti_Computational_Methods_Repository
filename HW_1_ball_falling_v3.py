import numpy as np
import matplotlib.pyplot as plt
import argparse as ap

parser = ap.ArgumentParser(description='calculate and plot the time, position, and speed of a ball in free fall')
parser.add_argument('height', type=float, help='This is the height from which the ball is being dropped')
parser.add_argument('--acceleration', type=float, default = 9.81, help='This is the acceleration due to gravity')
args = parser.parse_args()

t = np.linspace(0,100,20) #time from 0 to 10 seconds

def ball_in_freefall(height, acceleration):
    speed = acceleration*t #speed starting at 0, with acceleration due to gravity
    position = height-0.5*acceleration*np.square(t) #position with position starting at zero, calculates position at 20 different timepoints
    #acceleration in a data array
    a = np.full((20, 1),acceleration)
    t_to_fall = np.sqrt(2 * height / acceleration)
    return a, speed, position, t_to_fall

def plot_freefall(height, acceleration):
    #all three kinematic variables

    #acceleration plot
    fig, axs = plt.subplots(3, 1, figsize=(7, 7))
    axs[0].plot(t, ball_in_freefall(height, acceleration)[0], color = "purple")
    axs[0].set_ylim(0, acceleration+5)
    axs[0].set_xlim(0, ball_in_freefall(height, acceleration)[3])
    axs[0].set_ylabel('acceleration [m/s/s]')
    axs[0].grid(True)
    axs[0].set_title('A Ball Falling Down')

    #speed plot
    axs[1].plot(t, ball_in_freefall(height, acceleration)[1], color = "green" )
    axs[1].set_ylim(0, acceleration*(ball_in_freefall(height, acceleration)[3])+5)
    axs[1].set_xlim(0, ball_in_freefall(height, acceleration)[3])
    axs[1].set_ylabel('speed [m/s]')
    axs[1].grid(True)

    #position plot
    axs[2].plot(t, ball_in_freefall(height, acceleration)[2], color = "darkblue")
    axs[2].set_ylim(0, height+5)
    axs[2].set_xlim(0, ball_in_freefall(height, acceleration)[3])
    axs[2].set_xlabel('time [s]')
    axs[2].set_ylabel('position [m]')
    axs[2].grid(True)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    print(ball_in_freefall(args.height,args.acceleration)[3], "seconds to hit the ground")

plot_freefall(args.height,args.acceleration)
