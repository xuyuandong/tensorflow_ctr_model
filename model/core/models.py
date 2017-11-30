from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf

#from productnet import utils
import utils
from utils import _variable_with_weight_decay, _variable_on_cpu
import batch_input
dtype = utils.DTYPE

FLAGS = tf.app.flags.FLAGS


def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.B = None
        self.vars = None
        self.layer_keeps = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self, sess, fetches, X=None, B=None, y=None, mode='train'):
            feed_dict = {}
            if type(self.X) is list:
                for i in range(len(X)):
                    feed_dict[self.X[i]] = X[i]
            else:
                feed_dict[self.X] = X
            if B is not None:
                feed_dict[self.B] = B
            if y is not None:
                feed_dict[self.y] = y
            if self.layer_keeps is not None:
                if mode == 'train':
                    feed_dict[self.layer_keeps] = self.keep_prob_train
                elif mode == 'test':
                    feed_dict[self.layer_keeps] = self.keep_prob_test
            return sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)


class LR(Model):
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', 
            learning_rate=1e-2, l2_weight=0, sync=False, workers=20):
        Model.__init__(self)

        #self.graph = tf.Graph()
        #with self.graph.as_default():
        with tf.device('/cpu:0'):
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.vars = utils.init_var_map(init_vars, init_path)
        w = self.vars['w']
        b = self.vars['b']
        
        xw = tf.sparse_tensor_dense_matmul(self.X, w)
        logits = tf.reshape(xw + b, [-1])
        self.y_prob = tf.sigmoid(logits)

        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)

        self.global_step = _variable_on_cpu('global_step', [], 
                    initializer=tf.constant_initializer(0), trainable=False)
        if sync: 
            self.optimizer = utils.get_sync_optimizer(opt_algo, learning_rate, workers)
        else:
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate)

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class FM(Model):
    def __init__(self, data_dir=None, eval_dir=None, summary_dir=None, num_epochs=1,
            batch_size=None, input_dim=None, output_dim=1, factor_order=10, init_path=None,
            opt_algo='gd', learning_rate=1e-2, l2_w=0, l2_v=0, sync=False, workers=20):
        Model.__init__(self)

        data_file_list = tf.gfile.ListDirectory(data_dir)
        data_file_list = [x for x in data_file_list if '.tf' in x];
        data_file_list = [os.path.join(data_dir, x) for x in data_file_list]
        data_file_list.sort()
        eprint("input files:", data_file_list)
        input_files = data_file_list
        
        eprint("-------- create graph ----------")
        #self.graph = tf.Graph()
        #with self.graph.as_default():
        with tf.device('/cpu:0'):
            self.X = tf.sparse_placeholder(tf.float32, name='X')
            self.B = tf.sparse_placeholder(tf.float32, name='B')
            self.y = tf.placeholder(tf.float32, shape=[None], name='y')
                
        init_vars = [('linear', [input_dim, output_dim], 'xavier', dtype),
                     ('V', [input_dim, factor_order], 'xavier', dtype),
                     ('bias', [output_dim], 'zero', dtype)]
        
        self.vars = utils.init_var_map(init_vars, None)
        w = self.vars['linear']
        V = self.vars['V']
        b = self.vars['bias']
                
        ## linear term
        Xw = tf.sparse_tensor_dense_matmul(self.B, w)
                
        ## cross term
        # X^2
        X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
        # XV, shape: input_dim*k
        XV_square = tf.square(tf.sparse_tensor_dense_matmul(self.X, V))
        # X^2 * V^2, shape: input_dim*k
        X2V2 = tf.sparse_tensor_dense_matmul(X_square, tf.square(V))
        
        ## normalize
        Xnorm = tf.reshape(1.0/tf.sparse_reduce_sum(self.X, 1), [-1, output_dim])
        # 1/2 * row_sum(XV_square - X2V2), shape: input_dim*1
        p = 0.5 *Xnorm * tf.reshape(tf.reduce_sum(XV_square - X2V2, 1), [-1, output_dim])
                
        logits = tf.reshape(b + Xw + p, [-1])
                
        self.y_prob = tf.sigmoid(logits)
                
        self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(Xw)
        self.global_step = _variable_on_cpu('global_step', [], 
                    initializer=tf.constant_initializer(0), trainable=False)
        if sync: 
            self.optimizer = utils.get_sync_optimizer(opt_algo, learning_rate, workers)
        else:
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate)

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()
            


class FMUV(Model):
    def __init__(self, data_dir=None, summary_dir=None, eval_dir=None, 
            batch_size=None, input_dim=None, output_dim=1, factor_order=10, init_path=None,
            opt_algo='gd', learning_rate=1e-2, l2_w=0, sync=False, workers=20):
        Model.__init__(self)

        eprint("-------- create graph ----------")
        with tf.name_scope('input_%d' % FLAGS.task_index) as scope:
            self.X = tf.sparse_placeholder(tf.float32, name='X')
            self.B = tf.sparse_placeholder(tf.float32, name='B')
            self.y = tf.placeholder(tf.float32, shape=[None], name='y')
            
        init_vars = [('linear', [input_dim, output_dim], 'xavier', dtype),
                     ('U', [input_dim, factor_order], 'xavier', dtype),
                     ('V', [input_dim, factor_order], 'xavier', dtype),
                     ('bias', [output_dim], 'zero', dtype)]
        
        self.vars = utils.init_var_map(init_vars, None)
        w = self.vars['linear']
        U = self.vars['U']
        V = self.vars['V']
        b = self.vars['bias']
        
        ## normalize
        Xnorm = tf.reshape(1.0/tf.sparse_reduce_sum(self.X, 1), [-1, output_dim])
        
        ## linear term
        Xw = tf.sparse_tensor_dense_matmul(self.B, w, name="Xw")

        ## cross term
        XU = tf.sparse_tensor_dense_matmul(self.X, U, name="XU")
        XV = tf.sparse_tensor_dense_matmul(self.X, V, name="XV")
        X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X))) 
        p = 0.5 * Xnorm * tf.reshape(
                    tf.reduce_sum(XU*XV - tf.sparse_tensor_dense_matmul(X_square, U*V), 1),
                    [-1, output_dim])
            
        logits = tf.reshape(b + Xw + p, [-1])
            
        self.y_prob = tf.sigmoid(logits)
        #
        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(Xw)
            
        self.global_step = _variable_on_cpu('global_step', [], 
                initializer=tf.constant_initializer(0), trainable=False)

        if sync: 
            self.optimizer = utils.get_sync_optimizer(opt_algo, learning_rate, workers)
        else:
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate)
        
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()


class FNN(Model): #deepFM
    def __init__(self, data_dir=None, summary_dir=None, eval_dir=None, 
            batch_size=None, input_dim=None, output_dim=1, layer_sizes=None, layer_acts=None, 
            drop_out=None, init_path=None, opt_algo='gd', learning_rate=1e-2, l2_w=0, layer_l2=None, 
            sync=False, workers=20):
        Model.__init__(self)
        
        eprint("-------- create graph ----------")
        
        init_vars = []
        
        # linear part
        init_vars.append(('linear', [input_dim, output_dim], 'xavier', dtype))
        init_vars.append(('bias', [output_dim], 'zero', dtype))
        
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # field_sizes[i] stores the i-th field feature number
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'xavier', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        
        # full connection
        node_in = num_inputs * factor_order
        init_vars.append(('w1', [node_in, layer_sizes[2]], 'xavier', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i+1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        
        #self.graph = tf.Graph()
        #with self.graph.as_default():
        #with tf.device('/cpu:0'):
        with tf.name_scope('input_%d' % FLAGS.task_index) as scope:
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.B = tf.sparse_placeholder(tf.float32, name='B')
            self.y = tf.placeholder(dtype)
           
        self.keep_prob_train = 1 - np.array(drop_out)
        self.keep_prob_test = np.ones_like(drop_out)
        self.layer_keeps = tf.placeholder(dtype)
            
        self.vars = utils.init_var_map(init_vars, init_path)
        w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
        b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
        xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
        x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
        
        ## normalize
        fmX = tf.sparse_add(self.X[0], self.X[1])
        for i in range(2, num_inputs):
            fmX = tf.sparse_add(fmX, self.X[i])
        Xnorm = tf.reshape(1.0/tf.sparse_reduce_sum(fmX, 1), [-1, output_dim])
        
        l = tf.nn.dropout(utils.activate(x, layer_acts[0]), self.layer_keeps[0])

        for i in range(1, len(layer_sizes) - 1):
            wi = self.vars['w%d' % i]
            bi = self.vars['b%d' % i]
            eprint(l.get_shape(), wi.get_shape(), bi.get_shape())
            l = tf.nn.dropout(
                    utils.activate(tf.matmul(l, wi) + bi, layer_acts[i]),
                    self.layer_keeps[i])

        ## FM linear part
        fmb = self.vars['bias']
        fmw = self.vars['linear']
        Xw = tf.sparse_tensor_dense_matmul(self.B, fmw)
        ## cross term
        # XV, shape: input_dim*k
        fmXV = tf.add_n(xw)
        XV_square = tf.square(fmXV)
        eprint(XV_square.get_shape())
        # X^2 * V^2, shape: input_dim*k
        fmX2 = [tf.SparseTensor(self.X[i].indices, tf.square(self.X[i].values),
                tf.to_int64(tf.shape(self.X[i]))) for i in range(num_inputs) ]
        fmV2 = [tf.square(w0[i]) for i in range(num_inputs) ]
        fmX2V2 = [tf.sparse_tensor_dense_matmul(fmX2[i], fmV2[i]) for i in range(num_inputs) ]
        X2V2 = tf.add_n(fmX2V2)
        eprint(X2V2.get_shape())
        
        # 1/2 * row_sum(XV_square - X2V2), shape: input_dim*1
        p = 0.5 * Xnorm * tf.reshape(tf.reduce_sum(XV_square - X2V2, 1), [-1, output_dim])

        ## logits
        logits = tf.reshape(l + Xw + fmb + p, [-1])
        ## predict
        self.y_prob = tf.sigmoid(logits)

        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                l2_w * tf.nn.l2_loss(Xw)
        if layer_l2 is not None:
            self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw,1))
            for i in range(1, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
        
        self.global_step = _variable_on_cpu('global_step', [], 
                initializer=tf.constant_initializer(0), trainable=False)
        
        if sync: 
            self.optimizer = utils.get_sync_optimizer(opt_algo, learning_rate, workers)
        else:
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate)
        
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()


class CCPM(Model):
    def __init__(self, field_sizes=None, embed_size=10, filter_sizes=None, layer_acts=None, drop_out=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        init_vars.append(('f1', [embed_size, filter_sizes[0], 1, 2], 'xavier', dtype))
        init_vars.append(('f2', [embed_size, filter_sizes[1], 2, 2], 'xavier', dtype))
        init_vars.append(('w1', [2 * 3 * embed_size, 1], 'xavier', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embed_size, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    int(num_inputs / 2)),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                utils.activate(
                    tf.reshape(l, [-1, embed_size * 3 * 2]),
                    layer_acts[0]),
                self.layer_keeps[0])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.matmul(l, w1) + b1

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class PNN1(Model):
    def __init__(self, data_dir=None, summary_dir=None, eval_dir=None, 
            batch_size=None, input_dim=None, output_dim=1, layer_sizes=None, layer_acts=None, drop_out=None,
                 layer_l2=None, kernel_l2=None, l2_w=0, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 sync=False, workers=20):
        Model.__init__(self)
        
        eprint("------- create graph ---------------")

        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        
        with tf.name_scope('input_%d' % FLAGS.task_index) as scope:
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.B = tf.sparse_placeholder(tf.float32, name='B')
            self.y = tf.placeholder(dtype)
           
        self.keep_prob_train = 1 - np.array(drop_out)
        self.keep_prob_test = np.ones_like(drop_out)
        self.layer_keeps = tf.placeholder(dtype)
            
        self.vars = utils.init_var_map(init_vars, init_path)
        w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
        b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
        xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
        x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
        l = tf.nn.dropout(utils.activate(x, layer_acts[0]), self.layer_keeps[0])
            
        w1 = self.vars['w1']
        k1 = self.vars['k1']
        b1 = self.vars['b1']
        p = tf.reduce_sum(
                tf.reshape(
                    tf.matmul(
                        tf.reshape(
                            tf.transpose(
                                tf.reshape(l, [-1, num_inputs, factor_order]),
                                [0, 2, 1]),
                            [-1, num_inputs]),
                        k1),
                    [-1, factor_order, layer_sizes[2]]),
                1)
        l = tf.nn.dropout(
            utils.activate(
                    tf.matmul(l, w1) + b1 + p,
                    layer_acts[1])
               ,self.layer_keeps[1])

        for i in range(2, len(layer_sizes) - 1):
            wi = self.vars['w%d' % i]
            bi = self.vars['b%d' % i]
            l = tf.nn.dropout(
                utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i])
                    ,self.layer_keeps[i])


        ## logits
        l = tf.reshape(l, [-1])
        self.y_prob = tf.sigmoid(l)

        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            
        if layer_l2 is not None:
            self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw,1))
            for i in range(1, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
        if kernel_l2 is not None:
            self.loss += kernel_l2 * tf.nn.l2_loss(k1)
        
        self.global_step = _variable_on_cpu('global_step', [], 
            initializer=tf.constant_initializer(0), trainable=False)
        
        if sync: 
            self.optimizer = utils.get_sync_optimizer(opt_algo, learning_rate, workers)
        else:
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate)
        
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.summary_op = tf.summary.merge_all()


class PNN2(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None,
                 layer_norm=True):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        node_in = num_inputs * embed_size + embed_size * embed_size
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero',  dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)

            z = tf.reduce_sum(tf.reshape(xw, [-1, num_inputs, embed_size]), 1)
            op = tf.reshape(
                tf.matmul(tf.reshape(z, [-1, embed_size, 1]),
                          tf.reshape(z, [-1, 1, embed_size])),
                [-1, embed_size * embed_size])

            if layer_norm:
                # x_mean, x_var = tf.nn.moments(xw, [1], keep_dims=True)
                # xw = (xw - x_mean) / tf.sqrt(x_var)
                # x_g = tf.Variable(tf.ones([num_inputs * embed_size]), name='x_g')
                # x_b = tf.Variable(tf.zeros([num_inputs * embed_size]), name='x_b')
                # x_g = tf.Print(x_g, [x_g[:10], x_b])
                # xw = xw * x_g + x_b
                p_mean, p_var = tf.nn.moments(op, [1], keep_dims=True)
                op = (op - p_mean) / tf.sqrt(p_var)
                p_g = tf.Variable(tf.ones([embed_size**2]), name='p_g')
                p_b = tf.Variable(tf.zeros([embed_size**2]), name='p_b')
                # p_g = tf.Print(p_g, [p_g[:10], p_b])
                op = op * p_g + p_b

            l = tf.concat([xw, op], 1)
            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(tf.concat(w0, 0))
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
