#!/bin/bash
mkdir -p example/statlog_segment/r5/
./train train rf random example/statlog_segment/segment.dat example/statlog_segment/dataset_description.json example/statlog_segment/r5/ -n 5 -i 250 | tee example/statlog_segment/r5//train.log