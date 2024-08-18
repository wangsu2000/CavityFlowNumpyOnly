# CavityFlowNumpyOnly
2D Cavity Flow Simulation only depends on python numpy
The Source code and techniques are modified from 
https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver.git  

This work is a re-implement of matlab work.  

Anything about solving linear equations is replaced by numpy FFT solver  

And the stem of this work can be transplanted into cupy module.  

The 'Stable_CavityFlow_2or3order.py' gives the Crank-Nicolson(2th order) and RK3(3th order)
version in the original manuscripts while 'Euler_forward.py' gives a 1th order
numerical formulation.
The simulation streamlines are as follows:
![Streamlines](v2-dd18b58d2cf151602249ce0cc2560875_r.png)

A cupy version of data parallel version is released in gpuCavityFlow.py
The memory of Nvidia-gpu can be exploited efficiently by solving thousands 
of 2d equations simultaneously.

