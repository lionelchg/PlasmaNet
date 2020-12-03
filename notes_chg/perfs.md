## Plasma Oscillation
Running on 1 CPU node of krakengpu2
# Numpy Solver
```
Timer unit: 1e-06 s

Total time: 576.163 s
File: plasma_oscillation.py
Function: run at line 19

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    19                                           @profile
    20                                           def run(config):
    21                                               """ Main function containing initialization, temporal loop and outputs. Takes a config dict as input. """
    22         1      39766.0  39766.0      0.0      sim = PlasmaEuler(config)
    23                                               # Print header to sum up the parameters
    24         1       1511.0   1511.0      0.0      sim.write_init()
    25         1          1.0      1.0      0.0      if sim.verbose:
    26         1        130.0    130.0      0.0          sim.print_init()
    27                                           
    28                                               # Iterations
    29     10001      11452.0      1.1      0.0      for it in range(1, sim.nit + 1):
    30     10000       9615.0      1.0      0.0          sim.it = it
    31     10000      13427.0      1.3      0.0          sim.dtsum += sim.dt
    32     10000       9475.0      0.9      0.0          sim.time[it - 1] = sim.dtsum
    33                                                   
    34                                                   # Update of the residual to zero
    35     10000     310850.0     31.1      0.1          sim.res[:], sim.res_c[:] = 0, 0
    36                                           
    37                                                   # Solve poisson equation
    38     10000  346938412.0  34693.8     60.2          sim.solve_poisson()
    39                                           
    40                                                   # Compute euler fluxes (without pressure)
    41                                                   # sim.compute_flux()
    42     10000    2849680.0    285.0      0.5          sim.compute_flux_cold()
    43                                           
    44                                                   # Compute residuals in cell-vertex method
    45     10000  170585768.0  17058.6     29.6          sim.compute_res()
    46                                           
    47                                                   # Compute residuals from electro-magnetic terms
    48     10000    1270908.0    127.1      0.2          sim.compute_EM_source()
    49                                           
    50                                                   # boundary conditions
    51     10000      28148.0      2.8      0.0          sim.impose_bc_euler()
    52                                                   
    53                                                   # Apply residual
    54     10000     686641.0     68.7      0.1          sim.update_res()
    55                                           
    56                                                   # Post processing
    57     10000   49334670.0   4933.5      8.6          sim.postproc(it)
    58                                           
    59                                                   # Retrieve center variables 
    60     10000    3117193.0    311.7      0.5          sim.temporal_variables(it)
    61                                           
    62                                               # Plot temporals
    63         1     955279.0 955279.0      0.2      sim.post_temporal()

```
# Neural Network
```
Timer unit: 1e-06 s

Total time: 340.43 s
File: plasma_oscillation.py
Function: run at line 31

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    31                                           @profile
    32                                           def run(cfg_plasma, cfg_dl):
    33                                               """ Main function containing initialization, temporal loop and outputs. Takes a config dict as input. """
    34                                               
    35                                               # Load the network
    36         1        106.0    106.0      0.0      logger = cfg_dl.get_logger('test')
    37                                           
    38                                               # Setup data_loader instances
    39         1    1271328.0 1271328.0      0.4      data_loader = cfg_dl.init_obj('data_loader', module_data)
    40                                           
    41                                               # Build model architecture
    42         1      12817.0  12817.0      0.0      model = cfg_dl.init_obj('arch', module_arch)
    43                                           
    44                                               # Get function handles of loss and metrics
    45         1        495.0    495.0      0.0      loss_fn = cfg_dl.init_obj('loss', module_loss)
    46         1          7.0      7.0      0.0      metric_fns = [getattr(module_metric, metric) for metric in cfg_dl['metrics']]
    47                                           
    48         1        301.0    301.0      0.0      logger.info('Loading checkpoint: {} ...'.format(cfg_dl['resume']))
    49         1    7393104.0 7393104.0      2.2      checkpoint = torch.load(cfg_dl['resume'])
    50         1          5.0      5.0      0.0      state_dict = checkpoint['state_dict']
    51         1         14.0     14.0      0.0      if cfg_dl['n_gpu'] > 1:
    52                                                   model = torch.nn.DataParallel(model)
    53         1       5667.0   5667.0      0.0      model.load_state_dict(state_dict)
    54                                           
    55                                               # Prepare model for testing
    56         1         51.0     51.0      0.0      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    57         1       3043.0   3043.0      0.0      model = model.to(device)
    58         1        621.0    621.0      0.0      model.eval()    
    59                                               
    60         1      41170.0  41170.0      0.0      sim = PlasmaEulerDL(cfg_plasma)
    61                                               # Print header to sum up the parameters
    62         1      28653.0  28653.0      0.0      sim.write_init()
    63         1          1.0      1.0      0.0      if sim.verbose:
    64         1        105.0    105.0      0.0          sim.print_init()
    65                                           
    66                                               # Iterations
    67     10001      15114.0      1.5      0.0      for it in range(1, sim.nit + 1):
    68     10000      12244.0      1.2      0.0          sim.it = it
    69     10000      16053.0      1.6      0.0          sim.dtsum += sim.dt
    70     10000      12321.0      1.2      0.0          sim.time[it - 1] = sim.dtsum
    71                                                   
    72                                                   # Update of the residual to zero
    73     10000     313059.0     31.3      0.1          sim.res[:], sim.res_c[:] = 0, 0
    74                                           
    75                                                   # Solve poisson equation
    76     10000   41496355.0   4149.6     12.2          sim.solve_poisson_dl(model)
    77                                           
    78                                                   # Compute euler fluxes (without pressure)
    79     10000    5732015.0    573.2      1.7          sim.compute_flux_cold()
    80                                           
    81                                                   # Compute residuals in cell-vertex method
    82     10000  220457999.0  22045.8     64.8          sim.compute_res()
    83                                           
    84                                                   # Compute residuals from electro-magnetic terms
    85     10000    1256354.0    125.6      0.4          sim.compute_EM_source()
    86                                           
    87                                                   # boundary conditions
    88     10000      29894.0      3.0      0.0          sim.impose_bc_euler()
    89                                                   
    90                                                   # Apply residual
    91     10000     666697.0     66.7      0.2          sim.update_res()
    92                                           
    93                                                   # Post processing
    94     10000   57511280.0   5751.1     16.9          sim.postproc(it)
    95                                           
    96                                                   # Retrieve center variables 
    97     10000    3073621.0    307.4      0.9          sim.temporal_variables(it)
    98                                           
    99                                               # Plot temporals
   100         1    1079561.0 1079561.0      0.3      sim.post_temporal()

```