#! /bin/bash

if [ "$1" == "-h" ] || [ "$#" -ne 3 ]; then
	echo "Usage: `basename $0` logdir datadir dataset"
	exit 0
fi

LOGDIR=$1
DATADIR=$2
DATASET=$3

echo "$LOGDIR, $DATADIR, $DATASET"

for dirname in ${LOGDIR}/*; do
	if [ ! -d ${dirname}/metadata ]; then
		echo "Warning: no metadata/ directory at ${dirname}. Skipping..."
		continue
	fi
	# We've already computed it, so skip
	if [ -f ${dirname}/evaluation/summary_${DATASET}.txt ]; then
 	echo "Warning: Evaluation for ${dirname}/${DATASET} already complete. Skipping..."
		continue
	fi
  echo "Computing results for ${dirname}..."
	python scripts/evaluation/disentanglement.py compute --num_resamples 30 \
																						${dirname}/metadata \
																						${DATADIR} ${DATASET} \
																						${dirname}/evaluation/
	python scripts/evaluation/disentanglement.py summarize ${DATASET} ${dirname}/evaluation/ \
																		> ${dirname}/evaluation/summary_${DATASET}.txt
done
