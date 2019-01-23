#!/usr/bin/env sh
set -e
postfix=`date +"%F-%H-%M-%S"`
/home/yc/workplace/deeplearning/face/custom-caffe/build/tools/caffe train \
--solver=./solver.prototxt -gpu 0,1,2,3  \
--snapshot=./model/step2_1_iter_924000.solverstate \
2>&1 | tee ./Result/log/$(date +%Y-%m-%d-%H-%M.log) $@
