#!/bin/sh
###################################################
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
WK_DIR=`pwd`
cd ${WK_DIR}/script/
###################################################
source ${WK_DIR}/conf/default.conf


head="hdfs://${PROJECT}"
DICT_PATH="${head}/ctr_feature_dict"
TRAIN_INPUT="${head}/ctr_data_train/0*"
TRAIN_OUTPUT="${head}/ctr_data_train_libsvm"

PYLIBS=hdfs://user/xuyuandong/python_libs/Python.zip#Python

function libsvm() 
{
hdfs dfs -rmr $TRAIN_OUTPUT || true

hadoop jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar \
							 -D mapred.min.split.size=45000000000 \
							 -D mapreduce.map.speculative=false \
							 -D stream.job.map.memory.mb=4096 \
							 -D mapred.child.java.opts="-Xmx100m" \
							 -D mapreduce.task.timeout=600000000 \
							 -D mapreduce.job.reduces=100 \
							 -archives $PYLIBS \
							 -file $WK_DIR/encode/run_encoding.sh \
							 -file $WK_DIR/encode/rand_map.py \
							 -file $WK_DIR/encode/rand_reduce.py \
							 -input  $TRAIN_INPUT \
							 -output $TRAIN_OUTPUT \
							 -mapper "/bin/sh run_encoding.sh '$DICT_PATH'" \
							 -reducer "python rand_reduce.py"

}

function tfrecord()
{
TF_OUTPUT="${head}/ctr_data_train_tf"
hdfs dfs -rmr $TF_OUTPUT || true
hdfs dfs -mkdir -p $TF_OUTPUT || true
hdfs dfs -chmod 777 $TF_OUTPUT || true

OUTPUT=`date +"%Y%m%d_%H%M%S"`.`echo $$`
hdfs dfs -rmr -skipTrash $OUTPUT || true
hadoop jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar \
							 -D mapred.min.split.size=45000000000 \
							 -D mapreduce.map.speculative=false \
							 -D stream.job.map.memory.mb=4096 \
							 -D mapred.child.java.opts="-Xmx100m" \
							 -D mapreduce.task.timeout=600000000 \
							 -D mapreduce.job.reduces=0 \
							 -archives $PYLIBS \
							 -file $WK_DIR/encode/tfrecord_convert.py \
							 -file $WK_DIR/encode/run_tfrecord.sh \
							 -input  $TRAIN_OUTPUT \
							 -output $OUTPUT \
							 -mapper "/bin/sh run_tfrecord.sh '$TF_OUTPUT'" \
							 -reducer NONE
hdfs dfs -rmr -skipTrash $OUTPUT
}

libsvm
tfrecord
