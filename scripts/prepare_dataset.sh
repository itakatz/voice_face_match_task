#!/bin/bash

FACE_EMBED_FILE=vfm_assignment/image_embeddings.pickle
VOICE_EMBED_FILE=vfm_assignment/audio_embeddings.pickle
OUT_DATA_FOLDER=data

python prepare_dataset.py --face-embed-file $FACE_EMBED_FILE --voice-embed-file $VOICE_EMBED_FILE --out-data-folder $OUT_DATA_FOLDER
