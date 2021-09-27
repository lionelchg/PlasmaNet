# NeurIPS Workshop Machine Learning and the Physical Sciences submission

## Double headed streamer

### Without photoionization

To run the double headed streamer test case for a linear system solver and the neural network `UNet5-rfx800-rfy200` shown in Fig. 2 of the article:

```shell
run_cases -c dh_streamer_wo_photo.yml -np 2 -t streamer
```

Results are available in the `runs/dh_streamer_wo_photo/` directory. `case_1` corresponds to the neural network solver and `case_2` is the linear system solver for the Poisson equation. Figures can be found in the corresponding directories.

### With photoionization

To run the double headed streamer simulations with photoionization shown in Figures 2 and 4 of the article:

```shell
run_cases -c dh_streamer_photo.yml -np 4 -t streamer
```

4 simulations are run corresponding to the 4 different ways of solving the electric field/photoionization source terms. Results are available in the `runs/dh_streamer_photo/` directory and are described in the table below:

| Casename | Electric field | Photoionization |
| -------- | -------------- | --------------- |
| `case_1` | Linear system  | Linear system   |
| `case_2` | Linear system  | Neural network   |
| `case_3` | Neural network  | Linear system   |
| `case_4` | Neural network  | Neural network   |

Fig. 4 of the article corresponds to `case_4`.