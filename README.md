# Learning Interpretable and Disentangled Representations of Text using VAEs

### Requirements

`scikit-learn`
`pytorch`
`tqdm`
`numpy`


After editing `config_example.json` to your liking, run

```
python run.py config_example.json --verbose
```

Reconstructions are automatically logged under `logs/<experiment_name>/reconstructions_<dataset>.log`.
The script also automatically logs to Tensorboard. Start the server with

```
tensorboard --logdir runs
```

You can generate plots of the latent spaces using

```
python scripts/plot_zs.py zs_dir dataset latent_name outdir
```

For example, if your experiment is named `experiment1`,

```
python scripts/plot_zs.py logs/experiment1/metadata/zs train polarity logs/experiment1/metadata/plots
```

This produces a plot of the aggregated approximate posterior ∫q(z|x)p(x)dx for each dimension of the specified latent space.


### Reproducing Experiments

For every experiment run, the config file passed to `run.py` is logged to `logs/<experiment_name>/config_epoch<start_epoch>.json`.
You can use these config files directly to recreate an experiment. Besides some imprecision introduced by GPU computing,
and barring any fiddling with the source code, the experiments should recreate exactly. We recommend the following steps
to recreate an experiment named `experiment1`.

```
mkdir experiment1_configs/
cp logs/experiment1/config_*.json experiment1_configs/
vim experiment1_configs/*  # edit the "name" of each to be something like "experiment1_reprod"
# Run the experiments in order of `start_epoch`.
python run.py experiment1_configs/config_epoch0.json --verbose
.
.
.
python run.py experiment1_configs/config_epoch30.json --verbose
# etc.
```


### Sampling

Once a model has been trained, you can inspect it using the interactive sampling script.

```
python sample.py config.json
```

Type a sentence at the prompt to get a few reconstructions. Press ENTER with no input to generate a few sentences
randomly from the latent space.
