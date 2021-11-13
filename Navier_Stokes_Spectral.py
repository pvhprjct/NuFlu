# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 01:10:06 2021

@author: pvhprjct
"""

import numpy as np
import matplotlib.pyplot as plt

""" Fundamental parameters """
""" Refine mesh for smaller timesteps (dt) """

Nx = 64             # mesh size x -> resolution
Ny = 32             # mesh size y -> resolution
Lx = 2              # size of the container (x direction)
Ly = 1              # size of the container (y direction)
time = 0.5          # simulation's duration
dt = 10**(-3)       # timestep size -> delta t

dx = Lx/Nx          # number of discretization poins in space
dy = Ly/Ny
nt = time / dt      # number of discretization points in time (no. of timesteps)

Gr = 1e2            # Grashof number

""" Wave vectors (kappa) """

# 1D arrays for wave vectors ( to define the 2D arrays )
# No imaginary number in these arrays because the projection methods use the 
# wavevectors without an imaginary number
k_x = 2*np.pi*(np.fft.fftfreq(int(Lx/dx),d=dx))
k_y = 2*np.pi*(np.fft.fftfreq(int(Ly/dy),d=dy))

# 2D array placeholders for wave vectors

kx = np.zeros((Nx,Ny), dtype=complex)
ky = np.zeros((Nx,Ny), dtype=complex)
kxx = np.zeros((Nx,Ny), dtype=complex)
kyy = np.zeros((Nx,Ny), dtype=complex)

for i in range(len(k_x)):   # creates 2D arrays of (Ni,Ni) : i = {x,y}
    for j in range(len(k_y)):
        kx[i,j] = k_x[i]
        ky[i,j] = k_y[j]
        kxx[i,j] = k_x[i]**2
        kyy[i,j] = k_y[j]**2

k_squared = kxx + kyy # |k|^2

"""  Mesh """

x = np.arange(0, Lx, dx)     # x and y vectors to form the grid
y = np.arange(0, Ly, dy)

x_grid, y_grid = np.mgrid[0:Lx:1j*Lx/dx,0:Ly:1j*Ly/dy]  # 2D grid for plotting


"""
Initial conditions that satisfy the periodic conditions: 
"""

""" Velocity terms """

u = np.zeros((Nx,Ny))    # initial velocity in x direction
v = np.zeros((Nx,Ny))    # initial velocity in y direction

# if this is disabled, then the initial conditions are (u,v) = (0,0). Otherwise, look at the initial graph.
# I wrote it this way because it works.
for i in range(len(x)):
    for j in range(len(y)):
        u [i][j]= np.cos(2*np.pi * y[j] / Ly )* np.sin(2* np.pi * x[i] / Ly)
        v [i][j]= -np.sin(2* np.pi * y[j] / Ly) 

u_hat = np.fft.fft2(u) # Fourier transform of the ic in x direction -> corresponds to timestep n+1
u_hat_old = u_hat.copy()  # Corresponds to timestep n
u_hat_old_2 = u_hat_old.copy() # Corresponds to timestep n-1

v_hat = np.fft.fft2(v) # Fourier transform of the ic in y direction
v_hat_old = v_hat.copy()  # Corresponds to timestep n
v_hat_old_2 = v_hat_old.copy()  # Corresponds to timestep n-1

""" Forcing terms """

# Vorticity field 1
fx = ((2*np.pi)**3 / Gr) * ((Lx**2 + Ly**2) / Lx**2) * np.cos( 2 * np.pi * y_grid ) * np.sin( 2* np.pi * (Ly/Lx) * x_grid )
fy = -((2*np.pi)**3 / Gr) * ((Lx**2 + Ly**2) / Lx**2) * (Ly/Lx) * np.sin( 2 * np.pi * y_grid ) * np.cos( 2* np.pi * (Ly/Lx) * x_grid )

# Vorticity field 2
#fx = np.sin(2*np.pi*y_grid)
#fy = np.zeros((Nx,Ny))

fx_hat = np.fft.fft2(fx) # Fourier transform of the forcing terms
fy_hat = np.fft.fft2(fy)

""" Curl(v) in spectral space """
def omega_hat(u_hat,v_hat):   # Compute omega_hat
    return 1j * (kx * v_hat - ky * u_hat)

""" Projections in x- and y-directions """
def projection_x(gx, gy):
    return gx - (kx / k_squared) * ( kx * gx + ky * gy )

def projection_y(gx, gy):
    return gy - (ky / k_squared) * ( kx * gx + ky * gy )

""" Convolution product """

def convolution(u_input, v_input):
    w_input = omega_hat(u_input, v_input)  # Fourier transform of omega = curl(v)
    
    # transform w,u,v into real space
    w1_input, u1_input, v1_input = np.fft.ifft2(w_input), np.fft.ifft2(u_input), np.fft.ifft2(v_input)
    
    # calculate vw and uw in real space and transform to spectral space
    vw_output, uw_output = -np.fft.fft(v1_input * w1_input), np.fft.fft(u1_input * w1_input)
    
    return vw_output, uw_output

""" 
NS by spectral method - The periodic boundary conditions are implicit in the spectral method
"""

def timestep(u_hat, u_hat_old, u_hat_old_2, v_hat, v_hat_old, v_hat_old_2, fx_hat, fy_hat, Gr): 
    # '_hat' variables correspond to timestep n+1
    # '_hat_old' variables correspond to timestep n
    # '_hat_old_2' variables correspond to timestep n-1
    
    num = ( np.ones((Nx,Ny)) - (dt * k_squared)/2 ) # numerator constant
    den = ( np.ones((Nx,Ny)) + (dt * k_squared)/2 ) # denominator constant
    
    """ First step -> Crank-Nicolson + Explicit Euler step """
    
    """ Convolution product """

    vw_hat_old, uw_hat_old = convolution(u_hat_old, v_hat_old) 
        
    """ velocity fields """
    # x-direction    
    u_hat = ( num * u_hat_old - dt * ( projection_x(vw_hat_old, uw_hat_old) ) + dt * Gr * ( projection_x(fx_hat, fy_hat) ) ) / den
    u_hat[0 , 0] = 0    # Prevents division by zero (due to zeros in k_squared array)
    
    # y-direction
    v_hat = ( num * v_hat_old - dt * ( projection_y(vw_hat_old, uw_hat_old) ) + dt * Gr * ( projection_y(fx_hat, fy_hat) ) ) / den
    v_hat[0,0] = 0

    """ update variables """
    u_hat_old_2 = u_hat_old         #'_hat_old_2' terms are updated first. Otherwise, timestep n-1 would
    u_hat_old = u_hat               # equal n+1 with each iteration.
    
    v_hat_old_2 = v_hat_old    
    v_hat_old = v_hat
    
    """ Subsequent steps -> Crank-Nicolson + Adams-Bashforth of 2nd order """
        
    for i in range(0,int(nt)):
        
        """ Convolution product """
        vw_hat_old, uw_hat_old = convolution(u_hat_old, v_hat_old)  # timestep n
        
        vw_hat_old_2, uw_hat_old_2 = convolution(u_hat_old_2, v_hat_old_2)  # timestep n-1
        
        """ velocity fields """
        # x-component    
        u_hat = ( num * u_hat_old - 
                 (dt/2) * ( 3*projection_x(vw_hat_old, uw_hat_old) + projection_x(vw_hat_old_2, uw_hat_old_2) ) + 
                 dt * Gr * ( projection_x(fx_hat, fy_hat) ) ) / den
        u_hat[0,0] = 0 # Prevents division by zero
        
        # y-component
        v_hat = ( num * v_hat_old - 
                 (dt/2) * ( 3*projection_y(vw_hat_old, uw_hat_old) + projection_y(vw_hat_old_2, uw_hat_old_2)  ) +
                 dt * Gr * ( projection_y(fx_hat, fy_hat) ) ) / den
        v_hat[0,0] = 0
        
        """ update variables """
        u_hat_old_2 = u_hat_old         #'_hat_old_2' terms are updated first. Otherwise, timestep n-1 would
        u_hat_old = u_hat               # equal n+1 with each iteration.
        
        v_hat_old_2 = v_hat_old    
        v_hat_old = v_hat
        
    """ Compute dissipation """
    
    dissipation = ( np.real( np.fft.ifft2(vw_hat_old) )**2 ) + ( np.real( np.fft.ifft2(uw_hat_old) )**2 )
                
    return np.real(np.fft.ifft2(u_hat)), np.real(np.fft.ifft2(v_hat)), dissipation   # returns the real inverse Fourier transform - i.e. the solution


u_sol, v_sol, dissip = timestep(u_hat, u_hat_old, u_hat_old_2, v_hat, v_hat_old, v_hat_old_2, fx_hat, fy_hat, Gr)

E = 0.5*np.mean( u_sol * u_sol + v_sol * v_sol )
D = np.mean( dissip )


# loop to check for energy at different Gr
"""
t, E, D = [], [], []
for k in range(2,6):    
    t.append(k)
    Gr = 10**(k)
    u_sol, v_sol, dissip = timestep(u_hat, u_hat_old, u_hat_old_2, v_hat, v_hat_old, v_hat_old_2, fx_hat, fy_hat, Gr)
    
    E.append( 0.5*np.mean( u_sol * u_sol + v_sol * v_sol ) )
    D.append( np.mean( dissip ) )
"""    
    
print('Energy', E)
print('Dissipation', D)

""" Plot streamlines """

#plt.streamplot(y, x, u_sol, v_sol)  # solution
#plt.contourf(x_grid, y_grid, np.sin(u_sol)+np.cos(v_sol))

""" Plot heat map """
plt.imshow(np.transpose(u_sol))
plt.colorbar()

""" Plot labels """

CFL = u_sol.max()/dx + v_sol.max()/dy

print('$t$ for CFL = 1: $t =$  ', 1/CFL)

plt.ylabel('$u_{0x}$ Amplitude')
plt.xlabel('x')
plt.title('Navier-Stokes equations. Forcing term. \n $Gr = %1.0E$ \n Simulation time $= %1.0E$' %(Gr,time), pad = 20)

plt.show()

