import numpy as np
import matplotlib.pyplot as plt

height = float(input("Enter the height of the tower: ")) #user can set the height that the ball is falling
time = np.linspace(0,10,20) #time from 0 to 10 seconds
acceleration = 9.81 #acceleration due to gravity
speed = acceleration*time #speed starting at 0, with acceleration due to gravity
position = height-0.5*acceleration*np.square(time) #position with position starting at zero, calculates position at 20 different timepoints

#acceleration in a data array
a = np.full((20, 1),acceleration)

#all three kinematic variables
fig, axs = plt.subplots(3, 1, figsize=(7, 7))
axs[0].plot(time, a, color = "purple")
axs[0].set_ylabel('acceleration [m/s/s]')
axs[0].grid(True)
axs[0].set_title('A Ball Falling Down')

axs[1].plot(time, speed, color = "green" )
axs[1].set_ylabel('speed [m/s]')
axs[1].grid(True)

axs[2].plot(time, position, color = "darkblue")
axs[2].set_xlabel('time [s]')
axs[2].set_ylabel('position [m]')
axs[2].grid(True)

fig.tight_layout()
plt.show()
