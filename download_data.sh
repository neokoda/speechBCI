#!/bin/bash

# Stop on error
set -e

DATA_DIR="data"
mkdir -p $DATA_DIR

echo "Downloading Dataset from Dryad..."

# Direct download links from Dryad (based on file stream IDs)
# wget -O $DATA_DIR/sentences.tar.gz "https://datadryad.org/downloads/file_stream/2547372"
wget -O $DATA_DIR/languageModel.tar.gz "https://datadryad.org/downloads/file_stream/2547356"
wget -O $DATA_DIR/derived.tar.gz "https://datadryad.org/downloads/file_stream/2547370"

# Note: competitionData is a subset. If you are replicating the full paper, you want 'sentences'.
wget -O $DATA_DIR/competitionData.tar.gz "https://datadryad.org/downloads/file_stream/2547369"

echo "Extracting (assuming files are downloaded)..."

# if [ -f "$DATA_DIR/sentences.tar.gz" ]; then
#     tar -xzvf $DATA_DIR/sentences.tar.gz -C $DATA_DIR
# else
#     echo "sentences.tar.gz not found."
# fi

if [ -f "$DATA_DIR/competitionData.tar.gz" ]; then
    tar -xzvf $DATA_DIR/competitionData.tar.gz -C $DATA_DIR
else
    echo "competitionData.tar.gz not found."
fi

if [ -f "$DATA_DIR/languageModel.tar.gz" ]; then
    tar -xzvf $DATA_DIR/languageModel.tar.gz -C $DATA_DIR
else
    echo "languageModel.tar.gz not found."
fi

if [ -f "$DATA_DIR/derived.tar.gz" ]; then
    tar -xzvf $DATA_DIR/derived.tar.gz -C $DATA_DIR
else
    echo "derived.tar.gz not found."
fi

echo "Data setup (partial) complete. Please ensure files are downloaded."
