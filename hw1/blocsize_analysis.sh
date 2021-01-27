#!/bin/sh

echo "Beginning BLOCK_SIZE Analysis:"

for BLOCK_SIZE in {1..42}
do
  echo "BLOCK_SIZE = $BLOCK_SIZE"
  grep "Average percentage of Peak" temp/$BLOCK_SIZE/block_experiment_benchResults"$BLOCK_SIZE".txt
done
