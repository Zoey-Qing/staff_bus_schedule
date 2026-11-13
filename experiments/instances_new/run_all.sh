#!/bin/bash
# Wait for the long-search test to free the machine, then run the jobs four at a time.
until [ -f E:/schedule/revision/results_new/original/search_long/done.txt ]; do sleep 60; done
cd E:/schedule/revision/instances_new
cat jobs.txt | xargs -P 4 -L 1 ./pipeline.sh
echo ALL_DONE >> progress.txt
