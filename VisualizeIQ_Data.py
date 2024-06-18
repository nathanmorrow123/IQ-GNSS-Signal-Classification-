import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

# Define the directory containing the .mat files
workingdir = os.getcwd()
directory_path = os.path.join(workingdir, 'Raw_IQ_Dataset')
subFolder = os.path.join(directory_path, 'Training', 'NoJam')

# Define the sampling rate (replace with your actual sampling rate)
fs = 1e6  # Example: 1 MHz

# Initialize lists to store real and imaginary parts
all_real_parts = []
all_imaginary_parts = []

# Load each .mat file and extract the real and imaginary parts
for filename in sorted(os.listdir(subFolder))[0:30]:
    if filename.endswith('.mat'):
        file_path = os.path.join(subFolder, filename)
        mat_data = scipy.io.loadmat(file_path)
        varname = list(mat_data)[-1]
        iq_data = mat_data[varname].flatten()
        
        # Extract the real and imaginary parts
        real_part = np.real(iq_data)
        imaginary_part = np.imag(iq_data)
    
        all_real_parts.append(real_part)
        all_imaginary_parts.append(imaginary_part)

# Create the figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot([], [], label='Real Part')
line2, = ax.plot([], [], label='Imaginary Part (Shifted by 90°)', alpha=0.7)

ax.set_xlim(0, len(all_real_parts[0])/100)
ax.set_ylim(np.min([np.min(real) for real in all_real_parts]), np.max([np.max(real) for real in all_real_parts]))
ax.set_title('Real and Imaginary Parts of IQ Signal Over Time')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid()

# Initialize the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,

# Update function for the animation
def update(frame):
    line1.set_data(np.arange(len(all_real_parts[frame])), all_real_parts[frame])
    line2.set_data(np.arange(len(all_imaginary_parts[frame])), all_imaginary_parts[frame])
    return line1, line2,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(all_real_parts), init_func=init, blit=True, interval=100)

# Save the animation as an MP4 file
output_path = os.path.join(workingdir, 'iq_animation.gif')
ani.save(output_path, writer='ffmpeg')

# Show the animation
plt.show()
