#!/bin/bash

INPUT_DATA_FILE=data/data_num_neg_pp_2.pickle
RESULTS_FOLDER=saved_models_TMP/num_neg_pp_2
EPOCHS=100

python voice_face_match_train.py --input-data-file $INPUT_DATA_FILE --results-folder $RESULTS_FOLDER --epochs $EPOCHS
