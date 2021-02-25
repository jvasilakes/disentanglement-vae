# Learning Interpretable and Disentangled Representations of Text using VAEs

After editing `config_example.json` to your liking, run

```
python3 run.py config_example.json --verbose
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

This produce a plot of the approximate posterior q(z|x) for each dimension of the specified latent space.
