#!/bin/bash
DATASETS_DIR="external/datasets"
mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

# Get 50D GLoVE vectors
if hast wget 2>/dev/null; then
    wget http://nlp.stanford.edu/data/glove.6B.zip
else
    curl -L http://nlp.stanford.edu/data/glove.6B.zip -o glove.6B.zip
fi
unzip glove.6B.zip
rm glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt glove.6B.zip


