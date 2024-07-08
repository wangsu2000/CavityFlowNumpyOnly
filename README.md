# CavityFlowNumpyOnly
2D Cavity Flow Simulation only depends on python numpy
The Source code and techniques are modified from 
https://github.com/mathworks/2D-Lid-Driven-Cavity-Flow-Incompressible-Navier-Stokes-Solver.git \\
This work is a re-implement of matlab work. \\
Anything about solving linear equations is replaced by numpy FFT solver \\
And the stem of this work can be transplanted into cupy module. \\
The 'Stable_CavityFlow_2or3order.py' gives the Crank-Nicolson(2th order) and RK3(3th order)
version in the original manuscripts while 'Euler_forward_explicit.py' gives a 1th order
numerical formulation.
