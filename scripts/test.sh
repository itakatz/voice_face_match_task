#!/bin/bash

INPUT_TEST_DATA_FILE=test_data/test_data.pickle
MODEL_FILE=saved_models/num_neg_pp_5/best_model_epoch15.pth

python voice_face_match_test.py --input-test-file $INPUT_TEST_DATA_FILE --model-file $MODEL_FILE
