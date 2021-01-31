#!/bin/sh

echo "Beginning BLOCK_SIZE Optimization Trials (1-1000):"

echo "" > block_experiment_benchResults.txt
echo "" > block_experiment_compilerNotes.txt

if [ ! -d ./temp ];
then
  mkdir ./temp
fi
WORKING_DIR=$(pwd)

for i in {15..30}
do
  echo "BLOCK_SIZE = $i" 
  if [ ! -d ./temp/$i ];
  then
    mkdir ./temp/$i
  fi
  cp job-blocked ./temp/$i
  cp Makefile ./temp/$i
  cp *.c ./temp/$i
  cd ./temp/$i
  sbatch --export=BLOCK_SIZE=$i job-blocked
  cd $WORKING_DIR
# grep "Average percentage of Peak" block_experiment_benchResults"$BLOCK_SIZE".txt
done
