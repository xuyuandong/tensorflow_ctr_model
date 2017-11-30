from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys, os

import numpy as np
import tensorflow as tf

import batch_input
from models import FMUV, FNN, PNN1

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '.',
                            """ evaluate data file for local test """)
tf.app.flags.DEFINE_string('output_file', 'pred.txt', """ ouput """)

tf.app.flags.DEFINE_string('checkpoint_path', '',
                           """manual set which checkpoint to evaluate""")

tf.app.flags.DEFINE_integer('num_field', 50, """Number of feature field.""")
tf.app.flags.DEFINE_string('model', 'fnn', """CTR model.""")

tf.app.flags.DEFINE_integer('num_epochs', 1,
                            """Number of iteration for each worker.""")
tf.app.flags.DEFINE_integer('input_dim', 40000,
                            """Number of feature dimensional.""")
tf.app.flags.DEFINE_integer('batch_size', 5000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def worker_input(field_sizes=None):
    data_file_list = tf.gfile.ListDirectory(FLAGS.data_dir)
    data_file_list = [x for x in data_file_list if '.tf' in x];
    data_file_list = [os.path.join(FLAGS.data_dir, x) for x in data_file_list]
    data_file_list.sort()

    eprint(data_file_list)
    input_files = data_file_list

    X, y, B = batch_input.batch_input(input_files, batch_size=FLAGS.batch_size,
            shuffle=False, num_epochs=1, input_dim=FLAGS.input_dim, 
            field_sizes=field_sizes)
    return X, y, B

def eval_once(model, X, y, B):
    saver = tf.train.Saver(var_list=model.vars)

    pred_op = model.y_prob
    true_op = model.y
    loss_op = model.loss

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        saver.restore(sess, FLAGS.checkpoint_path)
        eprint('restore checkpoint: ', FLAGS.checkpoint_path)
        # extract global_step from it.
        global_step = FLAGS.checkpoint_path.split('/')[-1].split('-')[-1]
        eprint('global step = ', global_step)    

        fd = tf.gfile.Open(FLAGS.output_file, 'w')
        
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            step = 0
            loss_sum = 0    
            example_count = 0.01
            
            eprint("start predict ... ")
            while not coord.should_stop():
                step += 1
                _X, _y, _B = sess.run([X, y, B])
                loss, pred, true = model.run(sess, [loss_op, pred_op, true_op],
                        _X, _B, _y, mode='test') 
                loss_sum += np.sum(loss)
                example_count += len(pred) 
                true = np.reshape(true, (len(true),-1))
                pred = np.reshape(pred, (len(pred),-1))
                ret = np.concatenate((true, pred), axis=1)
                np.savetxt(fd, ret, fmt="%.1f %.6f")
                if step % 100 == 99:
                    eprint('examples=%d, loss_avg=%.3f' % (example_count, loss_sum/example_count))
        except Exception as e:    # pylint: disable=broad-except
            coord.request_stop(e)

        fd.close()
        eprint("final step = ", step)
        eprint("examples count  = ", example_count)
        eprint("step * batch_size = ", step * FLAGS.batch_size)
        eprint("loss avg  = ", loss_sum / example_count)
        
        
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):    # pylint: disable=unused-argument
    algo = FLAGS.model
    eprint(algo)
    field_sizes = None
    if algo == 'fmuv':
        params = {
            'data_dir': FLAGS.data_dir,
            'num_epochs': FLAGS.num_epochs,
            'batch_size': FLAGS.batch_size,
            'input_dim': FLAGS.input_dim,
            'factor_order': 12,
            'l2_w': 0.001,
        }
        eprint(params)
        model = FMUV(**params)
    elif algo == 'fnn':
        field_sizes = [FLAGS.input_dim] * FLAGS.num_field
        params = {
            'data_dir': FLAGS.data_dir,
            'batch_size': FLAGS.batch_size,
            'num_epochs': FLAGS.num_epochs,
            'input_dim': FLAGS.input_dim,
            'layer_sizes': [field_sizes, 12, 200, 1],
            'layer_acts': ['none', 'tanh', 'none'],
            'layer_l2': [0, 0, 0],
            'l2_w': 0.001,
        }
        eprint(params)
        model = FNN(**params)
    elif algo == 'pnn1':
        field_sizes = [FLAGS.input_dim] * FLAGS.num_field
        params = {
            'data_dir': FLAGS.data_dir,
            'batch_size': FLAGS.batch_size,
            'num_epochs': FLAGS.num_epochs,
            'input_dim': FLAGS.input_dim,
            'layer_sizes': [field_sizes, 12, 1],
            'layer_acts': ['tanh', 'none'],
            'layer_l2': [0, 0],
            'kernel_l2': 0,
            'l2_w': 0.001,
        }
        eprint(params)
        model = PNN1(**params)

    X, y, B = worker_input(field_sizes=field_sizes)
    eval_once(model, X, y, B)    






if __name__ == '__main__':
    tf.app.run()
