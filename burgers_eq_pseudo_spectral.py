# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:46:33 2021

@author: pvhprjct
"""
import numpy as np
import matplotlib.pyplot as plt

# Bigger N values prevent space instabilities
N, L, D, dt, time = 2*1024, 2*np.pi, 0.01, 10**(-4), 1.5 
step = 2*np.pi/N           # sampling interval

x = np.arange(0, L, step)  # grid (0,L) with N+1 steps of size 'step'
x0 = x.copy()

u = np.sin(x)              # initial condition
u_hat = np.fft.rfft(u)     # Fourier transform of the ic
u0_hat = u_hat.copy()

kappa = 2*np.pi*(np.fft.rfftfreq(int(L/step),d=step)*1j) # kappa and pre-factor 
kappa2 = kappa**2

# timestep
t_max = time/dt

def timestep(u_hat,u0_hat):         # u acts as u^{n+1} whereas u0 acts as u^{n}
    for i in range(0,int(t_max)):
        # For the quadratic term we transform back into physical space to compute the product.
        u1 = np.fft.irfft(u0_hat)
        uu = np.fft.rfft(u1**2)     # The result of the product is transformed back into spectral space.
        u_hat = u0_hat + dt*D*kappa2*u0_hat - dt*kappa*(uu)
        u0_hat = u_hat.copy()
        if (i*dt)%0.5==0 and (i*dt)!=0:
            u0 = np.fft.irfft(u0_hat)
            plt.plot( x,u0, label='$t = %1.1f$' %(i*dt) )
    return np.fft.irfft(u0_hat)     # returns real inverse Fourier transform

u0 = timestep(u_hat, u0_hat)
print(u0)

plt.plot( x,u0, label='$t = %1.1f$' %time )
    
plt.legend()
plt.title('Burgers equation - D = %1.2f' % D)
plt.ylabel('Temperature - $u(x,t)$')
plt.xlabel('Space')
plt.show()
