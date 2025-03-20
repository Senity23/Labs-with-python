#------------------------------------------------------------------------------
# Numerical solution using Finite Difference Method (Implicit scheme)
#------------------------------------------------------------------------------
# Time-stepping loop (keep other code same/similar to explicit scheme)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import time
#---------------------------------------------------------------------------
# Physical and computational parameters
#---------------------------------------------------------------------------
L = 1.0           # Length of the rod (meters)
T = 5             # Total simulation time (seconds)
N = 100           # Number of spatial grid points
M = 10000         # Number of time steps
alpha = 0.05      # Thermal diffusivity coefficient of material (m²/s)
dx = L / N        # Spatial step size (meters)
dt = T / M        # Time step size (seconds)
r = (alpha * dt / dx**2) 

assert r <= 0.5, "Stability condition violated! Decrease dt or increase dx."
# Create spatial grid
x = np.linspace(0, L, N+1)  # Equally spaced points from 0 to L
# Create temperature array: u[time_step, position]
u = np.zeros((M+1, N+1))  # M+1 time steps, N+1 spatial points
# Set initial conditions for rod temperature distribution
u[0, :] = np.sin(np.pi * x / L) * 50 + 20  # Temperatures from 20°C to 70°
# Jacobi iteration parameters
tol = 10^-6
max_iter = 100

for n in range(M):
    # Apply boundary conditions (your choice of Dirichlet or Neumann)
    u[n+1, 0] = u[n+1, 1]
    u[n+1, N] = u[n+1, N-1]

    
    # Initialize the solution for the current time step with previous values
    u[n+1, 1:N] = u[n, 1:N]

    # Create temporary arrays for Jacobi iteration
    u_old = np.copy(u[n+1, :])
    u_new = np.copy(u[n+1, :])
    
    # Iterative solution using Jacobi method
    for iter in range(max_iter):
        # Update interior points
        for i in range(1, N):
            u_new[i] = (u[n, i] + r * (u_old[i+1] + u_old[i-1])) / (1 + 2*r) 
        
        # Check convergence
        error = np.max(np.abs(u_new - u_old))
        if error < tol:
            break
            
        # Update old solution for next iteration 
        u_old = np.copy(u_new)
    
    # Update solution for current time step
    u[n+1, :] = u_new[:]
    
    # Print progress periodically
    if n % 100 == 0:
        print(f"Time step {n}/{M} completed ({n/M*100:.1f}%)")
#------------------------------------------------------------------------------
# Visualization 1: 1D Line Plot Animation
#------------------------------------------------------------------------------

# Clear everything before starting -----
plt.close('all')  # Close all existing figures

# Create figure and axis for the line plot
fig1 = plt.figure(figsize=(10, 6))
ax1 = plt.gca()

# Create initial line
line, = ax1.plot(x, u[0, :])
time_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

# Set up the plot
ax1.set_xlim(0, L)
ax1.set_ylim(0, 80)
ax1.set_xlabel('Position (m)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Heat Equation Solution with Dirichlet Boundaries (1D Rod)')
ax1.grid(True)

# Animation update function for line plot
def update_line(frame):
    n = frame * (M // 200)  # Show 200 frames total
    line.set_ydata(u[n, :])
    time_text1.set_text(f't = {n*dt:.2f} s')
    return line, time_text1

# Create the line plot animation
anim1 = FuncAnimation(
    fig1, 
    update_line,
    frames=200,
    interval=50,
    blit=True,
    repeat=True
)

# Save the line plot animation
if os.path.exists('heat_equation.gif'):
    try:
        os.remove('heat_equation.gif')
    except Exception as e:
        print(f"Warning: Could not remove existing heat_equation.gif file: {e}")

writer1 = PillowWriter(fps=20)
anim1.save('heat_equation.gif', writer=writer1)

#------------------------------------------------------------------------------
# Visualization 2: 2D Heatmap Animation
#------------------------------------------------------------------------------
# Create figure and axis with larger size for better visualization
fig2, ax2 = plt.subplots(figsize=(12, 4))

# Create a 2D representation of the rod by replicating the 1D temperature array
rod_thickness = 20  # Visual thickness of the rod (pixels)
rod_2d = np.tile(u[0, :], (rod_thickness, 1))  # Copy the 1D array vertically

# Create the image object for visualization
im = ax2.imshow(rod_2d, 
               aspect='auto',               # Adjust aspect ratio automatically
               extent=[0, L, -0.1, 0.1],    # Set the display limits [xmin, xmax, ymin, ymax]
               cmap='inferno',              # Use inferno colormap (good for temperature)
               vmin=20, vmax=70)            # Fix color scale to temperature range

# Add colorbar to show temperature scale
cbar = plt.colorbar(im)
cbar.set_label('Temperature (°C)', size=14)

# Set labels and title for the plot
ax2.set_xlabel('Position (m)', size=14)
ax2.set_title('Heat Equation Solution: Temperature Distribution in Rod', size=16, pad=20)

# Remove y-axis ticks as they're not meaningful for this visualization
ax2.set_yticks([])

# Add text to display current simulation time
time_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12, color='white')

# Animation update function for heatmap
def update_heatmap(frame):
    # Select time steps to visualize (we don't need to show all 10,000 steps)
    n = frame * (M // 200)  # Sample 200 frames from the full simulation
    
    # Update the 2D rod representation with current temperature profile
    rod_2d = np.tile(u[n, :], (rod_thickness, 1))
    im.set_array(rod_2d)
    
    # Update the time display
    time_text2.set_text(f't = {n*dt:.2f} s')
    
    # Return the objects that have been modified (required for blit=True)
    return im, time_text2

# Create the heatmap animation with 200 frames
anim2 = FuncAnimation(
    fig2,                # Figure to animate
    update_heatmap,      # Update function
    frames=200,          # Number of frames to display
    interval=50,         # Delay between frames in milliseconds
    blit=True,           # Use blitting for efficiency
    repeat=True          # Loop the animation
)

# Save the heatmap animation
if os.path.exists('heat_equation_rod.gif'):
    try:
        os.remove('heat_equation_rod.gif')
    except Exception as e:
        print(f"Warning: Could not remove existing heat_equation_rod.gif file: {e}")
        
writer2 = PillowWriter(fps=20)  # 20 frames per second
anim2.save('heat_equation_rod.gif', writer=writer2, dpi=150)  # Save with 150 dpi resolution

# Check if files exist and have content
if os.path.exists('heat_equation.gif') and os.path.getsize('heat_equation.gif') > 0:
    os.startfile('heat_equation.gif')  # Windows-specific command
else:
    print("Error: heat_equation.gif not found or empty")

# Small delay before opening second file to prevent overlap
time.sleep(0.5)

if os.path.exists('heat_equation_rod.gif') and os.path.getsize('heat_equation_rod.gif') > 0:
    os.startfile('heat_equation_rod.gif')  # Windows-specific command
else:
    print("Error: heat_equation_rod.gif not found or empty")

print("Animation files should now be open in your default GIF viewer")