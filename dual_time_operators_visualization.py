import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML

# Constants from the COM framework
LZ = 1.23498228  # Refined LZ constant
HQS = 0.235 * LZ  # Harmonic Quantum Scalar

# Time and frequency parameters
t = np.linspace(0, 2 * np.pi, 1000)
f_n = [1, 2.5, 4.3]  # Frequencies for the attractors
a_n = [1.0, 0.7, 0.5]  # Amplitudes
phi_n = [0, np.pi/4, np.pi/2]  # Phase shifts

def generate_recursive_signal(t, a_n, f_n, phi_n, coupling_factor):
    """Generate signal using recursive attractors modulated by coupling (cos(theta))."""
    signal = np.zeros_like(t)
    for a, f, p in zip(a_n, f_n, phi_n):
        signal += a * np.sin(f * t + p)
    return signal * coupling_factor

def generate_phase_gradient(t, coupling_factor):
    """Generate a phase gradient field based on COM principles."""
    gradient = np.gradient(np.sin(t * LZ) * np.cos(t * HQS))
    return gradient * coupling_factor

# Setup plot with GridSpec for multiple panels
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig)

# Time signals plot
ax1 = fig.add_subplot(gs[0, :])
line_wake, = ax1.plot([], [], lw=2, label="Waking Signal (T_ref)")
line_dream, = ax1.plot([], [], lw=2, label="Dream Signal (T_dream)", linestyle='--')
ax1.set_xlim(0, 2 * np.pi)
ax1.set_ylim(-2.5, 2.5)
ax1.set_title("Recursive Time Signal: Field-Coupled vs Internal Transformation")
ax1.set_xlabel("Phase (φ)")
ax1.set_ylabel("Signal Amplitude")
ax1.legend()
ax1.grid(True)

# Phase space plot
ax2 = fig.add_subplot(gs[1, 0])
phase_line, = ax2.plot([], [], lw=2, color='purple')
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-2.5, 2.5)
ax2.set_title("Phase Space Trajectory")
ax2.set_xlabel("Signal")
ax2.set_ylabel("Signal Derivative")
ax2.grid(True)

# Coupling visualization
ax3 = fig.add_subplot(gs[1, 1], projection='polar')
brain_vector, = ax3.plot([0, 0], [0, 1], lw=3, color='red', label="Brain Axis")
field_vector, = ax3.plot([0, 0], [0, 1], lw=3, color='blue', label="Field Normal")
ax3.set_title("Consciousness-Field Coupling")
ax3.set_rticks([0.5, 1.0])
ax3.set_rlim(0, 1.2)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Field tensor visualization
ax4 = fig.add_subplot(gs[2, :])
tensor_img = ax4.imshow(np.zeros((10, 10)), cmap='viridis', interpolation='nearest', 
                        extent=[-1, 1, -1, 1], origin='lower', aspect='auto')
ax4.set_title("Consciousness-Coupled Field Tensor")
ax4.set_xlabel("φ_μ")
ax4.set_ylabel("φ_ν")
fig.colorbar(tensor_img, ax=ax4, label="Tensor Magnitude")

# Equation display
equation_text = ax1.text(0.02, 0.92, "", transform=ax1.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))

# Add COM framework title
fig.suptitle("Dual Time Operators in the COM Framework", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

def init():
    line_wake.set_data([], [])
    line_dream.set_data([], [])
    phase_line.set_data([], [])
    brain_vector.set_data([0, 0], [0, 1])
    field_vector.set_data([0, 0], [0, 1])
    tensor_img.set_array(np.zeros((10, 10)))
    equation_text.set_text("")
    return line_wake, line_dream, phase_line, brain_vector, field_vector, tensor_img, equation_text

def animate(i):
    # Calculate angle and coupling
    angle_deg = i % 360
    angle_rad = np.radians(angle_deg)
    coupling = np.cos(angle_rad)  # Coupling strength C(x)
    
    # Generate signals
    signal_wake = generate_recursive_signal(t, a_n, f_n, phi_n, coupling)
    signal_dream = generate_recursive_signal(t, a_n, f_n, phi_n, 0.25)  # Weak coupling for dreams
    
    # Update time signals
    line_wake.set_data(t, signal_wake)
    line_dream.set_data(t, signal_dream)
    
    # Update phase space (signal vs. derivative)
    signal_derivative = np.gradient(signal_wake, t)
    phase_line.set_data(signal_wake[::10], signal_derivative[::10])  # Downsample for clarity
    
    # Update coupling visualization
    brain_vector.set_data([0, angle_rad], [0, 1])
    field_vector.set_data([0, 0], [0, 1])
    
    # Generate field tensor visualization
    phase_gradient = generate_phase_gradient(t, coupling)
    rho_E = 1.0  # Simplified energy density
    tensor = np.outer(phase_gradient[::100], phase_gradient[::100]) * rho_E * coupling
    tensor_img.set_array(tensor)
    
    # Update equation text based on mode
    if coupling > 0.7:  # Strong coupling - waking state
        eq_text = r"$T_{ref}(x,t) = \int \rho_E(x) \cdot \nabla\phi(x,t) \cdot \cos(\theta) dt$"
    else:  # Weak coupling - dream state
        eq_text = r"$T_{dream}(t) = \sum_n a_n \cdot \sin(f_n t + \phi_n)$"
    equation_text.set_text(eq_text)
    
    # Update titles with current values
    ax1.set_title(f"Recursive Time Signal — Field Angle: {angle_deg}°, Coupling: {coupling:.2f}")
    ax3.set_title(f"Consciousness-Field Coupling (C(x) = {coupling:.2f})")
    ax4.set_title(f"Consciousness-Coupled Field Tensor (T_μν^COM)")
    
    return line_wake, line_dream, phase_line, brain_vector, field_vector, tensor_img, equation_text

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=50, blit=True)

# Save animation
ani.save('/home/ubuntu/dual_time_operators.mp4', writer='ffmpeg', fps=30, dpi=100)

# Display final frame for reference
plt.savefig('/home/ubuntu/dual_time_operators_final.png', dpi=150)

print("Animation saved as 'dual_time_operators.mp4'")
print("Final frame saved as 'dual_time_operators_final.png'")
