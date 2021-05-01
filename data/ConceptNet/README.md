Data obtained from https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz
linked by https://github.com/norakassner/LAMA_primed_negated

`raw/test.jsonl` is the original data file included in the above `.tar.gz` file.

First, split this into train/dev/test split with
```
python split_polarity_data.py --infile raw/test.jsonl --outdir interim/conceptnet_split/
```

Then preprocess the data using
```
python get_polarity_data.py --indir interim/conceptnet_split/ --outdir processed/
```

Finally, specify `path/to/ConceptNet/processed/` in your config file to use the data.
