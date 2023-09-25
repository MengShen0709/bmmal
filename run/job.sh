#!/bin/bash

# before running this script, make sure you have copied initial AL iteration files to assure consistency

echo "config_file: $1"
echo "strategies: $2"
echo "random seed: $3"
echo "devices: $4"
config_file=$1
strategies=$2
seeds=$3
devices=$4

strategies=$(echo "$strategies" | tr "," "\n")
seeds=$(echo "$seeds" | tr "," "\n")

for seed in $seeds
do
  for round in {1..6}
  do
    for strategy in $strategies
    do
      echo "$config_file $strategy $seed $devices"
      python3 -u run/runner.py -c "$config_file" --strategy "$strategy" --seed "$seed" --devices "$devices" --round "$round"
    done
  done
done