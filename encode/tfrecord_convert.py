#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('label_type', 'float', '')
tf.app.flags.DEFINE_string('bias_fields', '', '')
tf.app.flags.DEFINE_string('model', '', '')

input_file = sys.argv[1]
output_file = sys.argv[2]

_int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
_float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))

bias_field_dict = {}
for key in FLAGS.bias_fields.split(','):
  bias_field_dict[key] = 1

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def convert_input_path(path, rindex, str):
    path = path.split('/')
    path[rindex] += str
    return "/".join(path)

def generate_simple_example(label, items, bias_field_dict):
    #row_indexes = []
    col_indexes = []
    col_values = []
    bias_indexes = []
    bias_values = []
    for item in items:
        field, col = item.split(':')
        #row_indexes.append(num)
        col_indexes.append(int(col))
        col_values.append(1.0)
        if field in bias_field_dict:
            bias_indexes.append(int(col))
            bias_values.append(1.0)

    label_ = _int_feature([label]) if FLAGS.label_type == 'int' else _float_feature([label])
    example = tf.train.Example(
                        features=tf.train.Features(
                                feature={
                                    'label': label_,
                                     #'row_index': _int_feature(row_indexes),
                                    'col_index': _int_feature(col_indexes),
                                    'col_value': _float_feature(col_values),
                                    'bias_index': _int_feature(bias_indexes),
                                    'bias_value': _float_feature(bias_values)

                                }))
    return example

def generate_field_example(label, items, bias_field_dict):
    col_indexes = {}
    col_values = {}
    bias_indexes = []
    bias_values = []
    for item in items:
        field, col = item.split(':')
        key = 'i_%s'%field
        val = 'v_%s'%field
        if key not in col_indexes:
            col_indexes[key] = [int(col)]
            col_values[val] = [1.0]
        else:
            col_indexes[key].append(int(col))
            col_values[val].append(1.0)
        if field in bias_field_dict:
            bias_indexes.append(int(col))
            bias_values.append(1.0)

    label_ = _int_feature([label]) if FLAGS.label_type == 'int' else _float_feature([label])
    
    feature = {'label': label_,
            'bias_idx': _int_feature(bias_indexes),
            'bias_val': _float_feature(bias_values)
            }
    for key in col_indexes:
        feature[key] = _int_feature(col_indexes.get(key, []))
    for val in col_values:  
        feature[val] = _float_feature(col_values.get(val, []))

    example = tf.train.Example(
                    features=tf.train.Features( feature=feature))
    return example

def main(argv=None):
    eprint("start convert ...")
    writer = tf.python_io.TFRecordWriter(output_file)
    num = 0
    for line in tf.gfile.Open(input_file):
        if num % 100000 == 0:
            eprint('%d lines done'%num)
            eprint("line=", line)
        
        l = line.rstrip().split()
        label = int(l[0]) if FLAGS.label_type == 'int' else float(l[0])
        label = 0 if label <= 0 else 1

        example = None
        if FLAGS.model == 'simple':
            example = generate_simple_example(label, l[1:], bias_field_dict)
        else:
            example = generate_field_example(label, l[1:], bias_field_dict)

        writer.write(example.SerializeToString())
        num += 1
    writer.close()

if __name__ == '__main__':
    tf.app.run()
