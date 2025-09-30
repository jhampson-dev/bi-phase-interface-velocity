This file allows for predicting the location of the fluid-fluid interface for bi-phase immiscible, incompressible, laminar flow in a striaght channel.
The user defines the parameters of channel height, the viscosities of both fluids and the pressure gradient driving flow. 

A function f(b) is defined with the analytical solution to the Navier-Stokes equation for incompressible, laminar flow in a straight channel (one of few analytical solutions to the Navier-Stokes equations). 
The interface height in the channel is b. Roots f(b) = 0 are found using brentq from the scipy.optimize package gives the region(s) where the interface can be found.

In the case of multiple valid roots f(b) = 0, this implies different modes of velocity profile can take shape in the channel. 
Which modes are more likely is beyond the scope of this file and likely requires stability analysis.

Plots show the user the interface position from the location of roots of f(b), the velocity as a function of channel height for their given parameters (presuming at least one root f(b) = 0 can be found), the (discontinuous if mu1 != mu2) transverse velocity gradient as a function of channel height and the continuous tangential stress as a function of channel height.
