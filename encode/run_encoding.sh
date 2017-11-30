#!/bin/bash

set -u

dict_dir=$1
echo "1: $dict_dir" >/dev/stderr

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

# download feature dictionary 
rm -rf dict.txt || true
hdfs dfs -getmerge ${dict_dir}/0* dict.txt || true
dictsize=$(get_filesize dict.txt)
echo "dict size: $dictsize" >/dev/stderr
if [ $dictsize -lt 10 ];then
  echo "failed to found encoding dict, bad exiting ..." >/dev/stderr
  cat >/dev/null
  exit 1
fi
echo "get dict successly" > /dev/stderr

# trainset format encoding ####################################################
echo "=== encoding data ===" >/dev/stderr
encode_cmd="TODO:command for encoding to libsvm using dict.txt"
echo "[shell cmd]: $encode_cmd" > /dev/stderr
$encode_cmd > stat_result 2> /dev/stderr
if [ $? -ne 0 ]; then
  echo 'ERROR happened in encoding data!' > /dev/stderr
  exit 1
fi

cat encoded_data | Python/bin/python rand_map.py

#############################################################################
echo "Encoding done." > /dev/stderr

exit 0
