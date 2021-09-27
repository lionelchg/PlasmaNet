# NeurIPS Workshop Machine Learning and the Physical Sciences submission

## Double headed streamer

The double headed streamer with photoionization can be run in 4 different ways

| Casename | Electric field | Photoionization |
| -------- | -------------- | --------------- |
| `case_1` | Linear system  | Linear system   |
| `case_2` | Linear system  | Neural network   |
| `case_3` | Neural network  | Linear system   |
| `case_4` | Neural network  | Neural network   |

```shell
run_cases -c double_headed_streamer.yml -np 4 -t streamer
```
