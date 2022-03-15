# Learning Disentangled and Informative Representations of Negation and Uncertainty


## Installation

```
git clone https://github.com/jvasilakes/disentanglement-vae.git
conda env create -f environment.yaml  # Install dependencies
conda activate dvae
python setup.py develop  # Install code under vae/
```

## Reproducing experiments from the paper

### Data

Tarballs of the raw data are stored in `data/tars/`. Follow the instructions in the associated READMEs (e.g. `data/SFU/README.md`) to preprocess them.

#### Combined SFU+Amazon
Unpack the SFU tarball at `data/tars/sfu_all.tar.gz` into `data/SFU/` and follow the instructions at `data/SFU/README.md`.
Unpack the Amazon tarball at `data/tars/amazon.tar.gz` into `data/Amazon/` and follow the instructions at `data/Amazon/README.md`.

Train the BOW negation and uncertainty classifiers on SFU.
```
python scripts/helpers/bow_classifier.py estimate data/SFU/processed data/SFU/bow_classifier_logs
```
Apply these models to the Amazon data to generate weak labels.
```
python scripts/helpers/bow_classifier.py apply data/SFU/bow_classifier_logs data/Amazon/processed data/Amazon/neg_unc_labels
```

Combine SFU with the labeled Amazon data
```
python scripts/helpers/combine_datasets.py --data_dirs data/SFU/processed data/Amazon/neg_unc_labels/ --dataset_names sfu amazon --Ns -1 100000 --outdir data/combined/sfu_amazon_100k/
```

`data/combined/sfu_amazon_100k/{train,dev,test}.jsonl` are now ready for training!
Make sure to set `"combined_dataset": true` and `"dataset_minibatch_ratios": {"sfu": float, "amazon": float}` (where the floats sum to 1) in the config file. The `dataset_minibatch_ratios` parameter specifies the percentage of examples in each minibatch that should be
taken from each dataset.


#### Combined SFU+Yelp
The process is the same as above using `data/tars/yelp.tar.gz` instead of the Amazon data.



### Model Training

Config files for each model reported in the paper are in `reproduction_configs`.

```
python run.py reproduction_configs/{model}.json [--verbose]
```

This command will train the model, run validation at every epoch, and run testing at the final epoch.
Log files, model checkpoints, and tensorboard files are by default saved to `$CWD/{logs,model_checkpoints,runs}/name/`
where `name` is whatever is set as the `name` field in the config file passed to `run.py`.

### Model Evaluation

All evaluation scripts are in `scripts/evaluation/`.

#### Disentanglement
using the combined SFU+Amazon dataset

```
# Compute the necessary statistics
python scripts/evaluation/disentanglement.py compute --num_resamples 30 logs/combined/sfu_amazon_100k/{model_name}/metadata/ data/combined/sfu_amazon_100k/ test logs/combined/sfu_amazon_100k/{model_name}/evaluation/

# Output the numbers, like table 1
python scripts/evaluation/disentanglement.py summarize test logs/combined/sfu_amazon_100k/{model_name}/evaluation/

# Plots MIG box plots, like figure 5
python scripts/evaluation/plot_migs.py logs/combined/sfu_amazon_100k/{first_model_name}/evaluation/MIGS_test.jsonl logs/combined/sfu_amazon_100k/{next_model_name}/evaluation/MIGS_test.jsonl ... outfile
```

#### Consistency
```
# Compute the necessary statistics.
python scripts/evaluation/consistency.py compute --num_resamples 30 logs/{model_name}/config_epoch0.json logs/{model_name}/evaluation/ test

Summarize the numbers, like table 4
python scripts/evaluation/consistency.py summarize logs/{model_name}/evaluation/ test
```

The script above will also, as a side-effect, compute self-BLEU scores (table 3) for each resample from the latent space. They are saved to `logs/{model_name}/evaluation/self_bleus_test.csv`. 


#### Invariance

*Details TBD, but the relevant latent parameters for computing correlations are logged under `logs/{model_name}/metadata/`*


#### Generation
*Details TBD*
```
python scripts/evaluation/compute_ppl.py
```
self-BLEUs at `logs/{model_name}/evaluation/self_bleus_*.csv`.


**Controlled Generation**
```
python scripts/evaluation/controlled_generation.py
```

For computing the regression models over the latent spaces (like table 12)
```
python scripts/helpers/predict_ntokens.py --dataset test logs/{model_name}/metadata data/{dataset_name}/processed/
```


## Running your own experiments

An example config file is at `config_example.json`. Check out `vae.utils.validate_params()` for more documentation.
Some things to note, the value of the `latent_dims` parameter is a mapping from label names (in the training data)
to latent dimensions. The overall dimensionality of the model is specified using the `total` key. All other keys
should match label names in the dataset. The number left over after subtracting the dimensionality of all labeled
spaces from `total` is assigne to a generic, unsupervised "content" space.

The `lambdas` keyword specifies the KL divergence weight for each of the latent spaces specified under `latent_dims`.
Any latent_spaces without a corresponding entry in `lambdas` will fall back to the value of the `default` key.
Set a weight to `"cyclic"` to use cyclic KL annealing for that latent. It will complete 4 cycles of linear increase
with a ratio of 0.5.

For the most part, the hyperparameters given in `config_example.json` are a good starting point.
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
python scripts/evaluation/plot_zs.py --data_split {train,dev,test} /path/to/logdir/metadata/ /path/to/datadir/
```

This produces a plot of the aggregated approximate posterior ∫q(z|x)p(x)dx for each dimension of the specified latent space.
Warning, this plots all dimensions separately, so its only adviseable to run this for small latents.



### Inspecting a model

Once a model has been trained, you can inspect it using the interactive sampling script.

```
python inspect_model.py /path/to/config.json
```

See the documentation within the script for more information.
