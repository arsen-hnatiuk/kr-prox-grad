# kr-prox-grad
Implementation of the Kantorovich-Rubinstein Proximal Gradient Method.

To create a conda environment with the required dependencies, run the command
```
conda env create -f environment.yml
conda activate kr_prox_grad_env
```

To recreate the experiments from the paper, run
```
python test/bang_bang_control.py
```