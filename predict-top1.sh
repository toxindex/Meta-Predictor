#!/bin/bash

SRC_FILE=$1
OUT_PATH=$2
INPUT_FILE=$3


OUT_FILE1=$OUT_PATH'/SOM.txt'
OUT_FILE2=$OUT_PATH'/metabolite.txt'
OUT_FILE3=$OUT_PATH'/predict.csv'
onmt_translate -model ./model/SoM_identifier/model1.pt ./model/SoM_identifier/model2.pt ./model/SoM_identifier/model3.pt ./model/SoM_identifier/model4.pt -src $SRC_FILE -output $OUT_FILE1 -n_best 1 -beam_size 4 -verbose -min_length 5 -max_length 150 -replace_unk -seed 42 -gpu -1
onmt_translate -model ./model/metabolite_predictor/model1.pt ./model/metabolite_predictor/model2.pt ./model/metabolite_predictor/model3.pt ./model/metabolite_predictor/model4.pt ./model/metabolite_predictor/model5.pt -src $OUT_FILE1 -output $OUT_FILE2 -n_best 1 -beam_size 3 -verbose -min_length 5 -max_length 150 -replace_unk -seed 42 -gpu -1
python process_predictions.py -input_file $INPUT_FILE -predictions_file $OUT_FILE2 -output_file $OUT_FILE3 -predict_number 1 -visualise_molecules True


