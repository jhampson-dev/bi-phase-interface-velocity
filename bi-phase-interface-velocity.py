import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ----- This script allows for finding the interface location of a bi-phase immiscible laminar flow, the velocity profile of the bi-phase flow and 

# Parameters
# h = 1.0  # Channel half-height  dummy value
h = float(input("Enter the nromalised half-height of the straight channel in metres (This value can be a floating point number): "))
print(h)
# dpdx = -1.0  # Pressure gradient dummy value
dpdx = int(input("Enter the pressure gradient driving flow in Pascal (This value can be a floating point number): "))
print(dpdx)

# mu1 = 1.0  # Viscosity bottom phase dummy value
mu1 = float(input("Input the dynamic viscosity of fluid 1 in Pascal seconds: "))
# mu2 = 2 * mu1  # Viscosity top phase dummy value
mu2 = float(input("Input the dynamic viscosity of fluid 2 in Pascal seconds: "))

# Define the function f(b). The interface height in the channel is b and roots to equation f(b) = 0 are the valid interface positions.
def f(b, mu1=mu1, mu2=mu2, dpdx=dpdx, h=h):
    p1 = (1 / (2 * mu1)) * dpdx * (((h + b)**3) / 3 - (h**2) * (h + b))
    
    denom = (1 - (mu1 / mu2)) * b + (1 + (mu1 / mu2)) * h
    p2 = (((1 / (2 * mu1)) - (1 / (2 * mu2))) * dpdx * (h**2 - b**2)) / denom
    p2 *= ((h + b)**2) / 2 + h * (h + b)
    
    p3 = (1 / (2 * mu2)) * dpdx * (((h - b)**3) / 3 - (h**2) * (h - b))
    
    p4 = ((mu1 / mu2) * ((1 / (2 * mu1)) - (1 / (2 * mu2))) * dpdx * (h**2 - b**2)) / denom
    p4 *= ((h - b)**2) / 2 - h * (h - b)
    
    return p1 + p2 - p3 - p4

# Root finding - single root in [-h, h].
# Changed simply from pre-existing MATLAB code. There may be a more Python-y way to do this with scipy.
b_root = brentq(f, -h, h)
print("Root b =", b_root)

# Sampling the function over the interval [-h, h] to find all roots
b_vals = np.linspace(-h, h, 2000)
f_vals = f(b_vals)

# Search for intervals where the function changes sign
root_intervals = []
for i in range(len(b_vals) - 1):
    if np.sign(f_vals[i]) != np.sign(f_vals[i + 1]):
        root_intervals.append((b_vals[i], b_vals[i + 1]))
        

# Check if multiple root intervals were found
if len(root_intervals) == 0:
    print("No roots found. No bi-phase flow!")
if len(root_intervals) == 1:
    print("One root found")
if len(root_intervals) > 1:
    print("Multiple roots found! Instability in interface position likely.")

# Root finding with duplicate filtering
roots_found = []
tol = 1e-6  # Tolerance for duplicate detection

for (p, q) in root_intervals:
    try:
        root = brentq(f, p, q)
        # Check duplicates
        if all(abs(root - r) > tol for r in roots_found):
            roots_found.append(root)
    except ValueError:
        # Skip intervals causing issues (e.g., no root or singularities)
        pass

roots_found = np.array(roots_found)

print("Roots found:")
print(roots_found)

# ----- Compute Velocity profile

def compute_velocity_profile(h, dpdx, mu1, mu2, b):
    # Domain discretization
    y_common = np.linspace(-h, h, 200)
    y1 = y_common[y_common <= b]  # bottom phase (fluid 1)
    y2 = y_common[y_common >= b]  # top phase (fluid 2)

    # Build coefficient matrix M (4x4)
    M = np.array([
        [-h, 1, 0, 0],   # u1(-h) = 0
        [0, 0, h, 1],    # u2(h) = 0
        [b, 1, -b, -1],  # velocity continuity at y=b
        [mu1, 0, -mu2, 0] # stress continuity at y=b
    ])

    # Right-hand side vector
    rhs = np.array([
        -(1 / (2 * mu1)) * dpdx * h**2,                  # u1(-h) = 0
        -(1 / (2 * mu2)) * dpdx * h**2,                  # u2(h) = 0
        -dpdx * (b**2) * ((1 / (2 * mu1)) - (1 / (2 * mu2))), # velocity continuity term
        0                                                 # stress continuity
    ])

    # Solve linear system for unknowns A1, B1, A2, B2
    X = np.linalg.solve(M, rhs)
    A1, B1, A2, B2 = X

    # Velocity profiles
    u1 = (1 / (2 * mu1)) * dpdx * y1**2 + A1 * y1 + B1
    u2 = (1 / (2 * mu2)) * dpdx * y2**2 + A2 * y2 + B2

    # Velocity gradients
    du1dy = (1 / mu1) * dpdx * y1 + A1
    du2dy = (1 / mu2) * dpdx * y2 + A2

    # Tangential stress profiles
    mududy1 = mu1 * du1dy
    mududy2 = mu2 * du2dy

    # Concatenate results
    y_all = np.concatenate([y1, y2])
    u_all = np.concatenate([u1, u2])
    dudy_all = np.concatenate([du1dy, du2dy])
    mududy_all = np.concatenate([mududy1, mududy2])

    return y_all, u_all, dudy_all, mududy_all


# Plotting
plt.figure(1)
plt.plot(b_vals, f_vals, label='f(b)')
plt.axhline(0, color='k', linestyle='--', label='f(b)=0')
plt.scatter(roots_found, f(roots_found), color='red', s=80, zorder=5, label='Roots')
plt.xlabel('b')
plt.ylabel('f(b)')
plt.title('Root finding to identify interface position')
plt.legend(fontsize=12)
plt.grid(True)

plt.figure(2)
plt.clf()
for b in roots_found:
    y, u, _, _ = compute_velocity_profile(h, dpdx, mu1, mu2, b)
    
    plt.plot(y, u, label=f'b = {b:.4f}')
    
    u_max = np.max(u)
    id_max = np.argmax(u)
    y_at_u_max = y[id_max]
    
    # Vertical dashed line at y where u is max
    plt.plot([y_at_u_max, y_at_u_max], [np.min(u), u_max + 0.05 * u_max], 'k--')
    
    # Vertical dashed line at y = b (interface)
    plt.plot([b, b], [np.min(u), np.max(u)], 'r--')

plt.grid(True)
plt.xlabel('y, Channel span [-h, +h]')
plt.ylabel('u(y), fluid velocity')
plt.title('Velocity profile as a function of y')
plt.xlim([np.min(y), np.max(y)])
plt.ylim([np.min(u), np.max(u)])
plt.legend(fontsize=10)
plt.gca().tick_params(labelsize=14)

# -----

plt.figure(3)
plt.clf()
for b in roots_found:
    y, _, dudy, _ = compute_velocity_profile(h, dpdx, mu1, mu2, b)
    
    plt.plot(y, dudy, label=f'b = {b:.4f}')
    plt.plot([b, b], [np.min(dudy), np.max(dudy)], 'r--')

plt.xlabel('y')
plt.ylabel('du(y)/dy')
plt.title('Transverse velocity gradient as a function of y')
plt.legend(fontsize=10)
plt.grid(True)
plt.gca().tick_params(labelsize=14)

# -----

plt.figure(4)
plt.clf()
for b in roots_found:
    y, _, _, mududy = compute_velocity_profile(h, dpdx, mu1, mu2, b)
    
    plt.plot(y, mududy, label=f'b = {b:.4f}')
    plt.plot([b, b], [np.min(mududy), np.max(mududy)], 'r--')

plt.xlabel('y')
plt.ylabel('mu*du(y)/dy')
plt.title('Tangential stress in channel')
plt.legend(fontsize=10)
plt.grid(True)
plt.gca().tick_params(labelsize=14)

plt.show()

