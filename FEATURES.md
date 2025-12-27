### Planned:
- Implicit time stepping
- Periodic boundaries
- Solid cells (rigid body in flow)
- GPU capability
- Shallow water also in 1D
- switch from ddx-functions on Field-level to ddx-functions on cell-level to reduce number of for loops in parallel region
- automatic time stepping (CFL number)

### Implemented:
- Introduce "Prob" class that includes all fields and prob parms
- Boundary conditions within Field-Classes
- Staggering
