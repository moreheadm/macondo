#!/bin/sh

# Get the number of times from the first argument
num_times=$1

# Check if the argument is provided and is a positive integer
if [[ ! $num_times =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [num_times]"
  echo "num_times must be a positive integer."
  exit 1
fi

for ((i=0; i<num_times; i++))
do
    python src/train.py -i "data/cp${i}.pt" -s 1024 -b "data/rb1.pt" -o "data/cp$((i + 1)).pt" train
done

