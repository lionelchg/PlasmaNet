# Journal of Computational Physics submission models and configuration files

## Plasma oscillation

To run plasma oscillation for two periods for a linear system solver and the neural network `UNet5-rf200` shown in Fig. 17(b) of the article:

```shell
run_cases -c plasma_oscillation.yml -np 2 -t pleuler
```

Results are available in the `runs/plasma_oscillation/` directory. `case_1` corresponds to the neural network solver and `case_2` is the linear system solver for the Poisson equation.Figures can be found in the corresponding directories.

## Double headed streamer

To run the double headed streamer test case for a linear system solver and the neural network `UNet5-rfx800-rfy200` shown in Fig. 20 of the article:

```shell
run_cases -c double_headed_streamer.yml -np 2 -t streamer
```

Results are available in the `runs/double_headed_streamer/` directory. `case_1` corresponds to the neural network solver and `case_2` is the linear system solver for the Poisson equation. Figures can be found in the corresponding directories.

