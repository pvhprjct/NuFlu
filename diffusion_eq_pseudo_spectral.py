# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:46:33 2021

@author: pvghd
"""
import numpy as np
import matplotlib.pyplot as plt

N, L, D, dt, time = 64, 1., 1., 10**(-5), 1 # N must be a power of 2
step = L/N # sampling interval

x = np.arange(0, L, step) # grid (0,1) with N steps of size 'step'
# I can't use N+1 steps because the final grid (after fft procedure) doesn't match
x0 = x.copy()

u = np.sin(2*np.pi*x)  # initial condition
u_hat = np.fft.rfft(u) # Fourier transform of the ic
u0_hat = u_hat.copy()

kappa = (2*np.pi*(np.fft.rfftfreq(int(L/step),d=step)*1j))**2 # kappa and pre-factor 

# timestep
t_max = time/dt

def timestep(u_hat,u0_hat):   # u acts as u^{n+1} whereas u0 acts as u^{n}
    for i in range(0,int(t_max)):
        u_hat[0] = 0
        u_hat =  u0_hat + dt*D*kappa*u0_hat
        u_hat[-1] = 0
        u0_hat = u_hat.copy()
        if (i*dt)%0.02==0 and (i*dt)!=0:
            u0 = np.fft.irfft(u0_hat)
            plt.plot( x,u0, label='$t = %1.1f$' %(i*dt) )
    return np.fft.irfft(u0_hat)   # returns the real inverse Fourier transform - i.e. the solution

u0 = timestep(u_hat, u0_hat)
print(u0)

plt.plot( x,u0, label='$t = %1.1f$' %time )
    
plt.legend()
plt.title('Diffusion equation - D = %1.0f' % D)
plt.ylabel('Temperature - $u(x,t)$')
plt.xlabel('Space')
plt.show()