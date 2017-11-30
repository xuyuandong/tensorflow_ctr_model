#!/bin/sh
###################################################
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
WK_DIR=`pwd`
###################################################
source ${WK_DIR}/conf/default.conf

set -u
end_date=$1

app_name=tensorflow_model
eval_name=tensorflow_eval
model_name=deepfm

head="hdfs://${PROJECT}"
TEST_DATA="${head}/ctr_data_test_tf/*.tf"
MODEL_PATH="${head}/${app_name}/${model_name}/${end_date}"
EVAL_PATH="${head}/${eval_name}/${model_name}/${end_date}"
INFO_PATH="${EVAL_PATH}_info"

hdfs dfs -rmr ${EVAL_PATH}
hdfs dfs -mkdir -p ${EVAL_PATH}
hdfs dfs -chmod 777 ${EVAL_PATH}

hdfs dfs -rmr ${INFO_PATH}

cd ${WK_DIR}/model/
PYLIBS=hdfs://user/xuyuandong/python_libs/Python.zip#Python

hadoop jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar \
							 -D mapred.min.split.size=45000000000 \
							 -D mapreduce.map.speculative=false \
							 -D stream.job.map.memory.mb=4096 \
							 -D mapred.child.java.opts="-Xmx100m" \
							 -D mapreduce.task.timeout=600000000 \
							 -D mapreduce.job.reduces=0 \
							 -archives $PYLIBS \
                             -file $WK_DIR/model/core \
							 -input $TEST_DATA \
							 -output $INFO_PATH \
							 -mapper "/bin/sh core/run_eval.sh '$MODEL_PATH' '$EVAL_PATH'" \
							 -reducer NONE



