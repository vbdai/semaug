#!/usr/bin/env bash
echo "$0"
echo "$1"
echo "$2"

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/train_nocopy.py $1 --launcher pytorch ${@:3}
