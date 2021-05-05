# Learning Disentangled and Informative Representations of Negation and Uncertainty

### Requirements

See `requirements.txt`.

### Installation

```
python setup.py develop
```

An example config file is at `config_example.json`. Check out `utils.validate_params()` for more documentation.
Some things to note, the value of the `latent_dims` parameter is a mapping from label names (in the training data)
to latent dimensions. The overall dimensionality of the model is specified using the `total` key. All other keys
should match label names in the dataset. The number left over after subtracting the dimensionality of all labeled
spaces from `total` is assigne to a generic, unsupervised "content" space.

The `lambdas` keyword specifies the KL divergence weight for each of the latent spaces specified under `latent_dims`.
Any latent_spaces without a corresponding entry in `lambdas` will fall back to the value of the `default` key.

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
python scripts/evaluation/plot_zs.py zs_dir dataset latent_name outdir
```

For example, if your experiment is named `experiment1`,

```
python scripts/evaluation/plot_zs.py logs/experiment1/metadata/zs train polarity logs/experiment1/metadata/plots
```

This produces a plot of the aggregated approximate posterior ∫q(z|x)p(x)dx for each dimension of the specified latent space.
Warning, this plots all dimensions separately, so its only adviseable to run this for small latents.


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

Type a sentence at the prompt to get a few reconstructions, the norm of their representations,
and the latent space classifiers' predictions given these representations.
Press ENTER with no input to generate a few sentences randomly from the latent space.


### Measuring Disentanglement

The first script will estimate the Mutual Information Gap (MIG) for each latent space by sampling multiple
times from each space and then computing the mutual information between the latents and the ground truth generative factor values
(i.e. the labels). The second command will then summarize these results, printing out some stats and saving
some plots to `logs/<model_name>/evaluation`.

```
python scripts/evaluation/disentanglement.py compute logs/<model_name>/metadata data/<dataset>/processed {train,dev,test} logs/<model_name>/evaluation
python scripts/evaluation/disentanglement.py summarize {train,dev,test} logs/<model_name>/evaluation
```

### Encoder - Decoder Consistency
Consistency of the decoder and the encoder can be computed and measured with the following script.
Again, plots and detailed stats are saved to the evaluation directory.

```
python scripts/evaluation/decoding.py compute logs/<model_name>/<config_file>.json logs/<model_name>/evaluation {train,dev,test}
python scripts/evaluation/decoding.py summarize logs/<model_name>/evaluation {train,dev,test}
```

### Weak Supervision of Negation and Uncertainty

Since sentence level negation and uncertainty annotations are fairly straightforward, we leverage a simple Naive Bayes classifier
with BOW features to automatically annotate Amazon reviews. First, train and evaluate the classifier on SFU,

```
python scripts/helpers/bow_classifier.py estimate data/SFU/processed data/SFU/bow_classifier_logs
```

The models are saved at `data/SFU/bow_classifier_logs/models/`. The evaluation results are

```
$> cat data/SFU/bow_classifier_logs/results.log

uncertainty
Features:
['be' 'can' 'could' 'if' 'it' 'may' 'maybe' 'might' 'must' 'or' 'perhaps'
 'probably' 'seem' 'seemed' 'seems' 'should' 'think' 'want' 'would' 'you']
           precision  recall     F1        
train      0.9261     0.8949     0.9097    
dev        0.9131     0.8861     0.8990    

polarity
Features:
['ca' 'cannot' 'did' 'do' 'does' 'either' 'never' 'no' 'not' 'without']
           precision  recall     F1        
train      0.8943     0.8538     0.8727    
dev        0.9240     0.8534     0.8846    

```

Then apply this model on the Amazon data

```
python scripts/helpers/bow_classifier.py apply data/SFU/bow_classifier_logs data/Amazon/processed data/Amazon/neg_unc_labels
```

This command updates the Amazon dataset with labels and probabilities predicted by the classifier trained on SFU.

Now, since SFU and Amazon both have annotations for negation and uncertainty, we can combined them into a single dataset to
use in training. The following command puts all the SFU data and 100k examples from Amazon into a new dataset
at `data/combined/sfu_amazon_100k/`.

```
python scripts/helpers/combine_datasets.py --data_dirs data/SFU/processed data/Amazon/neg_unc_labels/ --dataset_names sfu amazon --Ns -1 100000 --outdir data/combined/sfu_amazon_100k/
```

Now we can train a VAE on the combined dataset by setting `"combined_dataset": true` and
`"dataset_minibatch_ratios": {"sfu": float, "amazon": float}` (where the floats sum to 1) in the config file.
The `dataset_minibatch_ratios` parameter specifies the percentage of examples in each minibatch that should be
taken from each dataset.
