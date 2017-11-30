from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf

def batch_input(filenames, input_dim=None, field_sizes=None, batch_size=None, num_epochs=None,
        shuffle=False, num_preprocess_threads=3, min_after_dequeue=None,
        random_seed=None, capacity=None, allow_smaller_final_batch=True):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    
    reader = tf.TFRecordReader()
    _, serialized_single_example = reader.read(filename_queue)

    if not min_after_dequeue: min_after_dequeue = 6*batch_size
    if not capacity: capacity = min_after_dequeue + 6*batch_size

    if shuffle == True:
        serialized_batch_example = tf.train.shuffle_batch(
            [serialized_single_example],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            allow_smaller_final_batch=allow_smaller_final_batch)
    else:
        serialized_batch_example = tf.train.batch(
            [serialized_single_example],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            allow_smaller_final_batch=allow_smaller_final_batch)

    if field_sizes is None:
        return parse_simple_example(serialized_batch_example, input_dim)
    else:
        return parse_field_example(serialized_batch_example, input_dim, field_sizes)

def parse_simple_example(serialized_batch_example, input_dim):
    data = tf.parse_example(
            serialized_batch_example,
            features={
                'label': tf.FixedLenFeature([],tf.float32),
                'deep': tf.SparseFeature(index_key='col_index',
                                       value_key='col_value',
                                       dtype=tf.float32,
                                       size=input_dim),
                'wide': tf.SparseFeature(index_key='bias_index',
                                       value_key='bias_value',
                                       dtype=tf.float32,
                                       size=input_dim)
    }) 
    return data['deep'], data['label'], data['wide']

def parse_field_example(serialized_batch_example, input_dim, field_sizes):
    features = {
            'label': tf.FixedLenFeature([], tf.float32),
            'wide': tf.SparseFeature(index_key='bias_idx', value_key='bias_val',
                                     dtype=tf.float32, size=input_dim)
            }
    num_field = len(field_sizes)
    for i in range(num_field):
        features['%d'%i] = tf.SparseFeature(index_key='i_%d'%(i),value_key='v_%d'%(i),
                                            dtype=tf.float32, size=field_sizes[i])
    
    data = tf.parse_example(serialized_batch_example, features = features)
    
    X = [ data['%d'%i] for i in range(num_field) ]
    return X, data['label'], data['wide']

