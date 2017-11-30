from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import sys, traceback
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
import batch_input
import utils
from models import FMUV, FNN, PNN1

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

eprint("========tensorflow version======", tf.__version__)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '.', """data dir""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/TF_distribute_train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/TF_distribute_eval',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('resume_dir', '', """data dir""")

tf.app.flags.DEFINE_boolean('sync_replicas', False,
                           """sync replicas optimizer.""")
tf.app.flags.DEFINE_integer('all_workers', 20,
                            """distributed workers for training.""")

tf.app.flags.DEFINE_string('log_dir', '',
                           """log dir.""")
tf.app.flags.DEFINE_integer('max_models_to_keep', 20,
                            """Max checkpoint models to keep.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Max global batch steps for the cluster learning.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                            """Learning rate for optimizer search.""")
tf.app.flags.DEFINE_string('optimizer', 'adagrad',
                            """Optimizer search method.""")
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            """Number of iteration for each worker.""")
tf.app.flags.DEFINE_integer('num_field', 55, """Number of feature field.""")
tf.app.flags.DEFINE_string('model', 'fnn', """CTR model.""")
tf.app.flags.DEFINE_integer('input_dim', 40000,
                            """Number of feature dimensional.""")
tf.app.flags.DEFINE_integer('batch_size', 10000,
                            """Number of batch size to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs, you can also specify ps:[1-5].example.co")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs, you can also specify worker[1-5].example.co")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")


def get_hosts(host_str):
  # handle the host contains [1-10] correctly
  arr = []
  for s in host_str.split(","):
    range_str = re.findall('\[.*?\]',s)
    if len(range_str) > 0:
      range_str = range_str[0]
      range_str = range_str[1:len(range_str)-1]
      prefix = s[:s.find('[')]
      suffix = s[s.find(']')+1:]
      lower = range_str.split("-")[0]
      higher = range_str.split("-")[1]
      for index in xrange(int(lower), int(higher) + 1):
         new_host = prefix + str(index) + suffix
         arr.append(new_host)
    else:
      arr.append(s)
  return arr

def flags_check():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    if not FLAGS.train_dir:
        raise ValueError('Please supply a train_dir') 


def worker_input(field_sizes=None):
    data_file_list = tf.gfile.ListDirectory(FLAGS.data_dir)
    data_file_list = [x for x in data_file_list if '.tf' in x];
    data_file_list = [os.path.join(FLAGS.data_dir, x) for x in data_file_list]
    data_file_list.sort()

    eprint(data_file_list)
    input_files = data_file_list

    tf.set_random_seed(FLAGS.task_index)
    X, y, B = batch_input.batch_input(input_files, batch_size=FLAGS.batch_size,
            shuffle=True, num_epochs=FLAGS.num_epochs, input_dim=FLAGS.input_dim, 
            field_sizes=field_sizes)
    return X, y, B


def worker_process(cluster, server):
    # assign ops to local worker by default
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
        ps_num = cluster.num_tasks('ps')
        worker_num = cluster.num_tasks('worker')

        algo = FLAGS.model
        eprint(algo)
        field_sizes = None
        if algo == 'fmuv':
            params = {
                'data_dir': FLAGS.data_dir,
                'summary_dir': FLAGS.train_dir,
                'eval_dir': FLAGS.eval_dir,
                'random_seed': FLAGS.task_index,
                'batch_size': FLAGS.batch_size,
                'num_epochs': FLAGS.num_epochs,
                'input_dim': FLAGS.input_dim,
                'learning_rate': FLAGS.learning_rate,
                'opt_algo': FLAGS.optimizer, #'adagrad',
                'sync': FLAGS.sync_replicas,
                'workers': FLAGS.all_workers,
                'factor_order': 12,
                'l2_w': 0.001,
            }
            eprint(params)
            model = FMUV(**params)
        elif algo == 'fnn':
            field_sizes = [FLAGS.input_dim] * FLAGS.num_field
            params = {
                'data_dir': FLAGS.data_dir,
                'summary_dir': FLAGS.train_dir,
                'eval_dir': FLAGS.eval_dir,
                'random_seed': FLAGS.task_index,
                'batch_size': FLAGS.batch_size,
                'num_epochs': FLAGS.num_epochs,
                'input_dim': FLAGS.input_dim,
                'learning_rate': FLAGS.learning_rate,
                'opt_algo': FLAGS.optimizer, #'adagrad',
                'sync': FLAGS.sync_replicas,
                'workers': FLAGS.all_workers,
                'layer_sizes': [field_sizes, 12, 200, 1],
                'layer_acts': ['none', 'tanh', 'none'],
                'drop_out': [0, 0, 0],
                'layer_l2': [0, 0, 0],
                'l2_w': 0.001,
            }
            eprint(params)
            model = FNN(**params)
        elif algo == 'pnn1':
            field_sizes = [FLAGS.input_dim] * FLAGS.num_field
            params = {
                'data_dir': FLAGS.data_dir,
                'summary_dir': FLAGS.train_dir,
                'eval_dir': FLAGS.eval_dir,
                'random_seed': FLAGS.task_index,
                'batch_size': FLAGS.batch_size,
                'num_epochs': FLAGS.num_epochs,
                'input_dim': FLAGS.input_dim,
                'learning_rate': FLAGS.learning_rate,
                'opt_algo': FLAGS.optimizer, #'adagrad',
                'sync': FLAGS.sync_replicas,
                'workers': FLAGS.all_workers,
                'layer_sizes': [field_sizes, 12, 1],
                'layer_acts': ['tanh', 'none'],
                'layer_l2': [0, 0],
                'kernel_l2': 0,
                'l2_w': 0.001,
            }
            eprint(params)
            model = PNN1(**params)

    worker_device="/job:worker/task:%d" % FLAGS.task_index
    with tf.device(worker_device):
        X, y, B = worker_input(field_sizes=field_sizes)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    #summary_writer = tf.summary.FileWriter(FLAGS.log_dir, model.graph)
    saver = tf.train.Saver(var_list=model.vars, max_to_keep=FLAGS.max_models_to_keep)
    save_interval = 100 if FLAGS.model == "fmuv" else 600 

    def load_pretrained_model(sess):
        restore_file = tf.train.latest_checkpoint(FLAGS.resume_dir)
        eprint('restore:', restore_file)
        saver.restore(sess, restore_file)
    load_model_function = load_pretrained_model if FLAGS.resume_dir != '' else None


    is_chief = (FLAGS.task_index == 0)
    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                            logdir=FLAGS.train_dir,
                            saver=saver,
                            init_fn = load_model_function,
                            global_step=model.global_step,
                            save_model_secs=save_interval)

    retry_times = 0
    N_failed = 10
    while retry_times < N_failed:
        try:
            eprint('retry_times = %d' % (retry_times))
            startt = time.time()
            with sv.managed_session(master=server.target) as sess:
                eprint('------ start ------', datetime.now())
                if is_chief:
                    time.sleep(10)
                run_while_batch(sv, sess, model, X, y, B)
            sv.stop()
            eprint("------ end sv stop:", datetime.now())
            endt = time.time()
            if endt - startt > 300:
                retry_times = N_failed
            else:
                time.sleep(10)
                retry_times += 1
        except :
            traceback.print_exc()
            retry_times += 1
            time.sleep(10)
   
def run_while_batch(sv, sess, model, X, y, B):
    loop_num = 0
    loss_sum = 0
    example_count = 0
    N = 20
    M = 5
    idx = 1
    slide_loss = [0.0] * N
    slide_exam = [0.01] * N
    while not sv.should_stop():
        if loop_num == 0:
            eprint("go to while and begin to train...")
        start_data = time.time()
        _X, _B, _y = sess.run([X, B, y])
        startt = time.time()
        _, train_loss_avg, step = model.run(sess, 
                [model.train_op, model.loss, model.global_step], _X, _B, _y)
        endt = time.time()

        cur_batch_size = _y.size
        train_loss = train_loss_avg * cur_batch_size
        example_count += cur_batch_size
        loss_sum += train_loss
        
        if loop_num >= idx * M:
            idx += 1
            slide_loss[idx % N] = 0.0
            slide_exam[idx % N] = 0.0
        slide_loss[idx % N] += train_loss
        slide_exam[idx % N] += cur_batch_size

        if loop_num == idx * M - 1:
            iotime = startt - start_data
            duration = endt - startt
            examples_per_sec = cur_batch_size / duration
            sec_per_io = float(iotime)
            sec_per_batch = float(duration)
            slide_loss_avg = sum(slide_loss) / sum(slide_exam)
            format_str = ('%s: step=%d, examples=%d, batch_loss=%.3f, slide_loss=%.3f, all_loss=%.3f (%.1f exs/sec; %.1f sec/batch; %.1f sec/io)')
            eprint(format_str % (datetime.now(), step, example_count, train_loss_avg,\
                        slide_loss_avg, loss_sum/example_count, \
                        examples_per_sec, sec_per_batch, sec_per_io))
            
        loop_num += 1
        
        if step > FLAGS.max_steps:
            eprint("reach max step, break...")
            break

            

def train():
    flags_check()
    ps_hosts = FLAGS.ps_hosts.split(",")
    eprint("ps_hosts: ", ps_hosts)
    worker_hosts = FLAGS.worker_hosts.split(",")
    eprint("worker_hosts: ", worker_hosts)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    eprint("job_name: ", FLAGS.job_name, "task_index: ", FLAGS.task_index)
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        worker_process(cluster, server)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
