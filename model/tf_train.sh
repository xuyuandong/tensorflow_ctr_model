#!/bin/sh
###################################################
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
WK_DIR=`pwd`
###################################################
source ${WK_DIR}/conf/default.conf

set -u
end_date=$1

app_name=tensorflow_model
model_name=deepfm

head="hdfs://${PROJECT}"
TRAIN_DATA="${head}/ctr_data_train_tf"
MODEL_PATH="${head}/${app_name}/${model_name}/${end_date}"

function new_train() {
hdfs dfs -rmr ${MODEL_PATH}
hdfs dfs -mkdir -p ${MODEL_PATH}
hdfs dfs -chmod 777 ${MODEL_PATH}
}

new_train

cd ${WK_DIR}/model/core
PYLIBS=hdfs://user/xuyuandong/python_libs/Python.zip#Python

TensorFlow_Submit  \
--appName $app_name \
--archives=$PYLIBS \
--files=train.py,models.py,utils.py,batch_input.py \
--ps_memory=10000 \
--worker_memory=10000 \
--worker_cores 4 \
--num_ps 50 \
--num_worker 100 \
--data_dir=${TRAIN_DATA} \
--train_dir=${MODEL_PATH} \
--mode_local=true \
--command=Python/bin/python,train.py model=fnn batch_size=10000 learning_rate=0.0001 optimizer=adam max_steps=100000 num_epochs=20

# if for fmuv: ps=2
#--command=Python/bin/python,main_batch.py model=fmuv batch_size=10000 learning_rate=0.0001 optimizer=adam max_steps=100000 num_epochs=20





#--tensorboard=true \
