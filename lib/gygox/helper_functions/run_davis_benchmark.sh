#!/bin/bash
# Runs davis benchmark on the given segmentation maps and outputs it as
#   txt and html
# See: http://davischallenge.org/ and https://github.com/fperazzi/davis
# Example command (From python use os.system):
# ./run_davis_benchmark.sh /path/to/masks/folder /path/to/benchmark-out
# Inputs:
# $1 : path to davis repo (where the davis repo is installed)
# $2 : path to the davis python environment folder (e.g. home/ubuntu/.virtualenvs/davis)
# $3 : path to segmented masks directory
# $4 : path to output directory
# $5 : eval_set: ['test', 'gygo-test', 'train']. Default: 'test'

# Changing relative paths to full paths
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
PATH_TO_DAVIS_REPO=$(get_abs_filename $1)
PATH_TO_DAVIS_ENV=$(get_abs_filename $2)
PATH_TO_BINARY_FRAMES=$(get_abs_filename $3)
PATH_TO_OUTPUT_DIR=$(get_abs_filename $4)
if [ -z "$5" ]; then
    EVAL_SET=test # default value
else
    EVAL_SET=$5
fi

BASENAME_OF_PATH=$(basename $PATH_TO_BINARY_FRAMES)

cd ${PATH_TO_DAVIS_REPO}/python
export PYTHONPATH=$(pwd)/lib
# activate virtualenv
source ${PATH_TO_DAVIS_ENV}/bin/activate # todo: make this not hardcoded

# Prepare config.py
#if [ ${EVAL_SET} = "gygo-test" ]; then
#    cp ./lib/davis/config-gygo.py ./lib/davis/config.py
#else
#    cp ./lib/davis/config-default.py ./lib/davis/config.py
#fi

TEMP_H5_FOLDER="../data/DAVIS/Results/Evaluation/480p/"
mkdir -p ${TEMP_H5_FOLDER} # mkdir will throw an error if the folder already exists
python -u tools/eval.py --metrics=J ${PATH_TO_BINARY_FRAMES} ${TEMP_H5_FOLDER}
python -u tools/eval_view.py --eval_set=${EVAL_SET} --output_dir=${PATH_TO_OUTPUT_DIR} ${TEMP_H5_FOLDER}${BASENAME_OF_PATH}.h5
