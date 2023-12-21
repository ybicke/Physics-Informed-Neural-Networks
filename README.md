# Physics-Informed-Neural-Networks

The main objective of the project is to apply ML algorithms to solve various tasks
related to the preliminary design of a thermal energy storage. In the following Physics Informed Neural Networks (PINN) and Fourier Neural Operator (FNO) algorithms were used:

- PINNs for solving PDEs
- PINNs for PDE-Constrained Inverse Problem
- FNO for Time Series Forecasting

The thermal energy device is used in solar power plants to store thermal energy during the charging phase and
release it for production of electricity during the discharging phase. The thermal energy is stored due to the interaction of a fluid and a solid phase. During the charging state the fluid is injected at high temperature from one end of the storage and heats the solid up. In contrast, during the discharging phase the reverse process occurs: cold fluid flows from the opposite end and absorbs heat from the solid. Between charging and discharging idle phases take place, where no fluid enters the thermal storage.

Therefore, at any instant of time the thermal storage can be in one of the following states:

1. Charging;
2. Idle between charging and discharging;
3. Discharging;
4. Idle between discharging and charging;
   
Together the four states establish a cycle and the same process is repeated for several cycles until
the thermal storage reaches a periodic or stationary regime.


### Mathematical Model

The thermal storage is modeled by a cylinder with length L and diameter D and it is assumed that temperature variation occurs only along the axis of the cylinder (see Figure 1 for a schematic representation of the thermal storage). The temperature evolution of the solid and fluid phases, Ts and Tf , is described by a system of two linear reaction-convection-diffusion equations.

![image](https://github.com/ybicke/Physics-Informed-Neural-Networks/assets/80389002/0e30b948-09ce-4e9a-9a35-39faa128c179)

with ρ being the density of the phases, C the specific heat, λ the diffusivity, ε the solid porosity, uf
the fluid velocity entering the thermal storage and hv the heat exchange coefficient between solid
and fluid. The fluid velocity is assumed to be uniform along the cylinder and varying only in time:
uf = u during charging, u = 0 during idle and uf = −u during discharging, with u being a positive
constant.

The system of equations has to be augmented with suitable initial and boundary conditions:

![image](https://github.com/ybicke/Physics-Informed-Neural-Networks/assets/80389002/e4d20b00-78b0-48ad-b322-f41b97d81c24)

![image](https://github.com/ybicke/Physics-Informed-Neural-Networks/assets/80389002/72a4c4a2-2077-445a-b0e5-68df2307ee8f)




