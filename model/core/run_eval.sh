#!/bin/bash

set -u

model_dir=$1
output_dir=$2
echo "1: $model_dir" >/dev/stderr
echo "2: $output_dir" >/dev/stderr

WK_DIR=`pwd`

echo "workspace dir: $WK_DIR" > /dev/stderr
echo "mapred input file: $mapreduce_map_input_file" > /dev/stderr


ATTEMPT_ID=$mapreduce_task_attempt_id
echo "ATTEMPT_ID: ${ATTEMPT_ID}" >/dev/stderr

echo "[shell cmd]: ls -l" > /dev/stderr
ls -l > /dev/stderr

get_filesize(){
				declare -i size=`[ -f "$1" ] && ls -l --block-size=1 "$1"|awk -F " " '{print $5;}'`
				echo $size
}

get_hdfs_filesize(){
				declare -i size=`hadoop fs -ls $mapreduce_map_input_file | awk -F' ' '{print $5}'`
				echo $size
}


put_hdfs_file(){
				retry=0
				hdfs_dir=`dirname $2`
				hdfs dfs -mkdir -p $hdfs_dir || true
				while ! hdfs dfs -put -f $1 $2.${ATTEMPT_ID}; 
				do
					if [[ $((retry++)) -lt 5 ]];then
						echo "put $2.${ATTEMPT_ID} failed" > /dev/stderr
						sleep 1
						hdfs dfs -rmr -f $2.${ATTEMPT_ID} || true
					else
						echo "fail to put $2.${ATTEMPT_ID}" >/dev/stderr
						exit 0
					fi
				done

				echo "succ to put $2.${ATTEMPT_ID}" >/dev/stderr

			  hdfs dfs -rmr -f $2 || true
				if hdfs dfs -mv $2.${ATTEMPT_ID} $2;then
					echo "succ rename to $2" >/dev/stderr
				else
					echo "fail rename to $2" >/dev/stderr
				fi
}

# input data name
dindex=`echo $mapreduce_map_input_file | awk -F'/' '{print $(NF)}'`
echo "output_file: $dindex" > /dev/stderr

# download model
rm -rf ckpt || true
hdfs dfs -get ${model_dir} ckpt || true
echo "get model successly" > /dev/stderr

# input data encoding ####################################################
rm -rf data || true
mkdir data || true
hdfs dfs -get $mapreduce_map_input_file data/
cat >/dev/null

# predict output dir 
rm -rf output || true
mkdir output || true

# evaluate each checkpoint ###########################
for file in `ls ckpt/*.meta`
do
    echo $file >/dev/stderr
    model=`echo $file | awk -F'.meta' '{print $(NF-1)}'`
    mindex=`echo $model | awk -F'/' '{print $(NF)}'`
    Python/bin/python core/eval.py --data_dir=data --output_file=output/$mindex.$dindex --checkpoint_path=$model

    if [ $? -ne 0 ]; then
      echo "ERROR happened in evaluate data: $model" > /dev/stderr
      exit 1
    fi
done


echo "upload encoded data to $output_dir" >/dev/stderr
for file in `ls output/*.tf`
do
    filename=`echo $file | awk -F'/' '{print $(NF)}'`
    put_hdfs_file $file $output_dir/$filename
done

#############################################################################
echo "Evaluating done." > /dev/stderr

exit 0
