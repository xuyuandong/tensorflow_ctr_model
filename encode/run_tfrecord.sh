#!/bin/bash

set -u

output_dir=$1
echo "1: $output_dir" >/dev/stderr

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


output_file=`echo $mapreduce_map_input_file | awk -F'/' '{print $(NF)}'`
echo "output_file tmp:${output_file}" > /dev/stderr
output_file=${output_dir}/${output_file}
echo "output_file: $output_file" > /dev/stderr

cat >encoded_data
echo "convert libsvm to tfrecord" >/dev/stderr
Python/bin/python tfrecord_convert.py encoded_data encoded_data_tf
echo "upload encoded data to $output_file" >/dev/stderr
put_hdfs_file encoded_data_tf ${output_file}.tf


#############################################################################
echo "Encoding done." > /dev/stderr

exit 0
