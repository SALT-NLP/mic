#!/bin/bash

MODELS=$1
DATA=$2 
FORMAT=$3 
i=$4

# NOTE : Quote it else use array to avoid problems #
#FILES="output_10k/*Q_A_TARGET_rot*"

for f in $MODELS
do
  echo "python -m  rot_generation.generate_rots_decoder --model_directory ${f}/model --input ${DATA} --output ${f} --format_string ${FORMAT} --gpu $i --beams 3 &"
  python -m  rot_generation.generate_rots_decoder --model_directory "${f}/model" --input "${DATA}" --output "${f}" --format_string "${FORMAT}" --gpu $i --beams 3 &
  echo "python -m  rot_generation.generate_rots_decoder --model_directory ${f}/model --input ${DATA} --output ${f} --format_string ${FORMAT} --gpu $i --top_p 0.9 &"
  python -m  rot_generation.generate_rots_decoder --model_directory "${f}/model" --input "${DATA}" --output "${f}" --format_string "${FORMAT}" --gpu $i --top_p 0.9 &
  echo "python -m  rot_generation.generate_rots_decoder --model_directory ${f}/model --input ${DATA} --output ${f} --format_string ${FORMAT} --gpu $i &"
  python -m  rot_generation.generate_rots_decoder --model_directory "${f}/model" --input "${DATA}" --output "${f}" --format_string "${FORMAT}" --gpu $i &
  ((i+=1))
done