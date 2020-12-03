# cases_1
```
casename: 'runs/kraken/cases_1/'
description: 'Influence of nt_oscill'
mode: 'seq'
params/nt_oscill : [500, 1000, 2000, 4000, 6000, 8000]
```

The amplification is reduced as the timestep gets smaller.
One needs around 6000 timesteps/period to get almost no amplification
over 3 periods.


# cases_2
```
casename: 'runs/kraken/cases_2/'
description: 'Influence of the resolution at cst cfl'
mode: 'seq'
params/nnx: [101, 201, 401]
params/nny: [101, 201, 401]
params/nt_oscill: [1000, 2000, 4000]
```

The amplification is reduced as the resolution if refined (as expected).
Very small amplification at the finest resolution

# cases_3
```
casename: 'runs/kraken/cases_3/'
description: 'Influence of the amplitude of perturbation'
mode: 'seq'
params/n_pert: [1.0e+10, 1.0e+11, 1.0e+12, 1.0e+13]
```

There is no influence of the amplitude of the perturbation on the propagation
of the oscillation, the amplification is the same regardless of the amplitude 
of the perturbation.

# Target case for network
