# Journal of Computational Physics submission models and configuration files

## Plasma oscillation

To run plasma oscillation for two periods for a linear system solver and the neural network `UNet5-rf200` shown in Fig. 17(b) of the article:

```shell
run_cases -c plasma_oscillation.yml -np 2 -t pleuler
```

## Double headed streamer

To run the double headed streamer test case for a linear system solver and the neural network `UNet5-rfx800-rfy200` shown in Fig. 20 of the article:

```shell
run_cases -c double_headed_streamer.yml -np 2 -t streamer
```
