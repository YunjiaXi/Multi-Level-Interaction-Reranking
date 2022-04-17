import itertools

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
import numpy as np
from tensorflow.python.framework import dtypes

from tensorflow.python.util import nest
import heapq


def tau_function(x):
    return tf.where(x > 0, tf.exp(x), tf.zeros_like(x))


def attention_score(x):
    return tau_function(x) / tf.add(tf.reduce_sum(tau_function(x), axis=1, keepdims=True), 1e-20)


class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num,
                 hist_spar_num, hist_dens_num, profile_num, max_norm=None):
        # reset graph
        tf.reset_default_graph()

        # input placeholders
        with tf.name_scope('inputs'):
            self.itm_spar_ph = tf.placeholder(tf.int32, [None, max_time_len, itm_spar_num], name='item_spar')
            self.itm_dens_ph = tf.placeholder(tf.float32, [None, max_time_len, itm_dens_num], name='item_dens')
            self.usr_profile = tf.placeholder(tf.int32, [None, profile_num], name='usr_profile')
            self.usr_spar_ph = tf.placeholder(tf.int32, [None, max_seq_len, hist_spar_num], name='user_spar')
            self.usr_dens_ph = tf.placeholder(tf.float32, [None, max_seq_len, hist_dens_num], name='user_dens')
            self.seq_length_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
            self.hist_length_ph = tf.placeholder(tf.int32, [None, ], name='hist_length_ph')
            self.label_ph = tf.placeholder(tf.float32, [None, max_time_len], name='label_ph')
            self.time_ph = tf.placeholder(tf.float32, [None, max_seq_len], name='time_ph')
            self.is_train = tf.placeholder(tf.bool, [], name='is_train')


            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.max_time_len = max_time_len
            self.max_seq_len = max_seq_len
            self.hidden_size = hidden_size
            self.emb_dim = eb_dim
            self.itm_spar_num = itm_spar_num
            self.itm_dens_num = itm_dens_num
            self.hist_spar_num = hist_spar_num
            self.hist_dens_num = hist_dens_num
            self.profile_num = profile_num
            self.max_grad_norm = max_norm
            self.ft_num = itm_spar_num * eb_dim + itm_dens_num
            self.feature_size = feature_size

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size + 1, eb_dim],
                                           initializer=tf.truncated_normal_initializer)
            self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
            self.usr_spar_emb = tf.gather(self.emb_mtx, self.usr_spar_ph)
            self.usr_prof_emb = tf.gather(self.emb_mtx, self.usr_profile)

            self.item_seq = tf.concat(
                [tf.reshape(self.itm_spar_emb, [-1, max_time_len, itm_spar_num * eb_dim]), self.itm_dens_ph], axis=-1)

    def build_fc_net(self, inp, scope='fc'):
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_mlp_net(self, inp, layer=(500, 200, 80), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn')
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 2, activation=None, name='fc_final')
            score = tf.nn.softmax(final)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_logloss(self, y_pred):
        # loss
        self.loss = tf.losses.log_loss(self.label_ph, y_pred)
        self.opt()

    def build_mseloss(self, y_pred):
        self.loss = tf.losses.mean_squared_error(self.label_ph, y_pred)
        self.opt()

    def build_attention_loss(self, y_pred):
        self.label_wt = attention_score(self.label_ph)
        self.pred_wt = attention_score(y_pred)
        # self.pred_wt = y_pred
        self.loss = tf.losses.log_loss(self.label_wt, self.pred_wt)
        # self.loss = tf.losses.mean_squared_error(self.label_wt, self.pred_wt)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
                # self.loss += self.reg_lambda * tf.norm(v, ord=1)

        # self.lr = tf.train.exponential_decay(
        #     self.lr_start, self.global_step, self.lr_decay_step,
        #     self.lr_decay_rate, staircase=True, name="learning_rate")

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm is not None:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')

            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw


    def train(self, sess, batch_data, lr, reg_lambda, keep_prob=0.8):
        de_lb = np.array(batch_data[-1])
        de_lb[de_lb > 10] = 10
        batch_data[-1] = de_lb
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.usr_profile: batch_data[0],
            self.itm_spar_ph: batch_data[1],
            self.itm_dens_ph: batch_data[2],
            self.usr_spar_ph: batch_data[3],
            self.usr_dens_ph: batch_data[4],
            # self.label_ph: batch_data[-1],
            self.label_ph: batch_data[5],
            self.time_ph: batch_data[6],
            self.seq_length_ph: batch_data[7],
            self.hist_length_ph: batch_data[8],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.is_train: True,
        })
        return loss

    def eval(self, sess, batch_data, reg_lambda, keep_prob=1, no_print=True):
        # fi_mat, ii_mat = [], []
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.usr_profile: batch_data[0],
            self.itm_spar_ph: batch_data[1],
            self.itm_dens_ph: batch_data[2],
            self.usr_spar_ph: batch_data[3],
            self.usr_dens_ph: batch_data[4],
            # self.label_ph: batch_data[-1],
            self.label_ph: batch_data[5],
            self.time_ph: batch_data[6],
            self.seq_length_ph: batch_data[7],
            self.hist_length_ph: batch_data[8],
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.is_train: False,
        })
        return pred.reshape([-1, self.max_time_len]).tolist(), label.reshape([-1, self.max_time_len]).tolist(), loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)


class MIR(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num,
                 hist_spar_num, hist_dens_num, profile_num, max_norm=None, intra_list=True, intra_set=True,
                 set2list='SLAttention', loss='log', fi=True, ii=True, decay=True):
        super(MIR, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num, profile_num, max_norm)

        with tf.variable_scope('MIR'):
            self.istrain = tf.placeholder(tf.bool, [])
            self.item_mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32),
                                  axis=-1)

            # intra-set interaction
            if intra_set:
                with tf.variable_scope('cross_item'):
                    item_seq = self.item_seq
                    seq = self.multihead_attention(item_seq, item_seq, num_heads=1)
                    seq = tf.concat([seq, self.item_seq], axis=-1)
                    self.cross_item_embed = seq * self.item_mask
            else:
                self.cross_item_embed = self.item_seq


            self.user_seq = tf.concat([tf.reshape(self.usr_spar_emb, [-1, max_seq_len, hist_spar_num * eb_dim]),
                                       self.usr_dens_ph], axis=-1)
            # intra-list interaction
            if intra_list:
                outputs, _, _ = self.bilstm(tf.unstack(self.user_seq, max_seq_len, 1), hidden_size, scope='user_bilstm')
                seq_ht = tf.reshape(tf.stack(outputs, axis=1), (-1, max_seq_len, hidden_size * 2))
                usr_seq = tf.concat([seq_ht, self.user_seq], -1)
            else:
                usr_seq = self.user_seq

            # set2list interaction
            with tf.variable_scope('set2list'):
                if set2list == 'co-att':
                    usr_seq = self.user_seq
                    v, q = self.co_attention(self.cross_item_embed, usr_seq)
                    seq = tf.concat([v, q], axis=-1)
                    self.set2list_embed = seq * self.item_mask
                elif set2list == 'SLAttention':
                    v, q = self.SLAttention(self.cross_item_embed, usr_seq, self.itm_spar_emb, self.usr_spar_emb,
                                            fi, ii, decay)
                    seq = tf.concat([v, q], axis=-1)
                    self.set2list_embed = seq * self.item_mask
                else:
                    self.set2list_embed = self.user_seq

            # mlp
            # self.final_embed = tf.concat([self.item_seq, self.intra_item_embed, self.set2list_embed], axis=-1)
            self.final_embed = tf.concat([self.item_seq, self.set2list_embed], axis=-1)
            # self.final_embed = self.set2list_embed
            self.y_pred = self.build_mlp_net(self.final_embed)

            # loss
            if loss == 'list':
                self.build_attention_loss(self.y_pred)
            elif loss == 'mse':
                self.build_mseloss(self.y_pred)
            else:
                self.build_logloss(self.y_pred)

    def feed_forward_net(self, inp, d_ff=256, scope='ffn'):
        with tf.variable_scope(scope):
            d_ft = inp.get_shape()[-1]
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            tmp = bn1 + inp
            fc1 = tf.layers.dense(tmp, d_ff, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, d_ft, activation=None, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            bn2 = tf.layers.batch_normalization(inputs=dp2, name='bn2')
        return inp + bn2

    def co_attention(self, V, Q, scope='co_att'):
        with tf.variable_scope(scope):
            v_dim, q_dim = V.get_shape()[-1], Q.get_shape()[-1]
            v_seq_len, q_seq_len = V.get_shape()[-2], Q.get_shape()[-2]
            bat_size = tf.shape(Q)[0]
            w_b = tf.get_variable("w_b", [1, q_dim, v_dim], initializer=tf.truncated_normal_initializer)
            C = tf.matmul(Q, tf.matmul(tf.tile(w_b, [bat_size, 1, 1]), tf.transpose(V, perm=[0, 2, 1])))
            C = tf.tanh(C)

            w_v = tf.get_variable('w_v', [v_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            w_q = tf.get_variable('w_q', [q_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            hv_1 = tf.reshape(tf.matmul(tf.reshape(V, [-1, v_dim]), w_v), [-1, v_seq_len, v_seq_len])
            hq_1 = tf.reshape(tf.matmul(tf.reshape(Q, [-1, q_dim]), w_q), [-1, q_seq_len, v_seq_len])
            hq_1 = tf.transpose(hq_1, perm=[0, 2, 1])
            h_v = tf.nn.tanh(hv_1 + tf.matmul(hq_1, C))
            h_q = tf.nn.tanh(hq_1 + tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            a_v = tf.nn.softmax(h_v, axis=-1)
            a_q = tf.nn.softmax(h_q, axis=-1)
            self.a_v = a_v
            self.a_q = a_q
            v = tf.matmul(a_v, V)
            q = tf.matmul(a_q, Q)
        return v, q

    def SLAttention(self, V, Q, V_s, Q_s, fi=True, ii=True, decay=True, scope='fi_s2l'):
        with tf.variable_scope(scope):
            v_dim, q_dim = V.get_shape()[-1], Q.get_shape()[-1]
            v_seq_len, q_seq_len = V.get_shape()[-2], Q.get_shape()[-2]
            bat_size = tf.shape(Q)[0]

            # get affinity matrix
            if fi:
                self.w_b_fi = tf.get_variable("w_b_fi", [1, self.emb_dim, self.emb_dim],
                                              initializer=tf.truncated_normal_initializer)

                V_s = tf.reshape(V_s, [-1, self.max_time_len * self.itm_spar_num, self.emb_dim])
                Q_s = tf.reshape(Q_s, [-1, self.max_seq_len * self.hist_spar_num, self.emb_dim])
                C2 = tf.matmul(Q_s, tf.matmul(tf.tile(self.w_b_fi, [bat_size, 1, 1]), tf.transpose(V_s, perm=[0, 2, 1])))
                C2 = tf.layers.conv2d(tf.expand_dims(C2, -1), 1, self.hist_spar_num, strides=(self.hist_spar_num, self.itm_spar_num))
                C2 = tf.reshape(C2, [bat_size, q_seq_len, v_seq_len])
                self.fi_mat = C2
            if ii:
                self.w_b = tf.get_variable("w_b", [1, q_dim, v_dim], initializer=tf.truncated_normal_initializer)
                C1 = tf.matmul(Q, tf.matmul(tf.tile(self.w_b, [bat_size, 1, 1]), tf.transpose(V, perm=[0, 2, 1])))
                self.ii_mat = C1
                if fi:
                    C1 = C1 + C2
            else:
                C1 = C2

            if decay:
                # decay
                pos = tf.reshape(tf.tile(tf.expand_dims(self.time_ph, -1), [1, 1, v_seq_len]),
                                 [-1, q_seq_len, v_seq_len])
                usr_prof = tf.reshape(self.usr_prof_emb, [-1, self.profile_num * self.emb_dim])
                usr_prof = tf.layers.dense(usr_prof, 32, activation=tf.nn.relu, name='fc_decay1')
                theta = tf.layers.dense(usr_prof, 1, activation=tf.nn.relu, name='fc_decay2')
                self.decay_theta = tf.tile(tf.reshape(theta, [-1, 1, 1]), [1, q_seq_len, v_seq_len])
                pos_decay = tf.exp(self.decay_theta * pos)
                C = tf.tanh(C1 * pos_decay + C1)
            else:
                C = C1

            # attention map
            w_v = tf.get_variable('w_v', [v_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            w_q = tf.get_variable('w_q', [q_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            hv_1 = tf.reshape(tf.matmul(tf.reshape(V, [-1, v_dim]), w_v), [-1, v_seq_len, v_seq_len])
            hq_1 = tf.reshape(tf.matmul(tf.reshape(Q, [-1, q_dim]), w_q), [-1, q_seq_len, v_seq_len])
            hq_1 = tf.transpose(hq_1, perm=[0, 2, 1])
            h_v = tf.nn.tanh(hv_1 + tf.matmul(hq_1, C))
            h_q = tf.nn.tanh(hq_1 + tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            # h_v = tf.nn.tanh(tf.matmul(hq_1, C))
            # h_q = tf.nn.tanh(tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            a_v = tf.nn.softmax(h_v, axis=-1)
            a_q = tf.nn.softmax(h_q, axis=-1)
            self.a_v = a_v
            self.a_q = a_q
            v = tf.matmul(a_v, V)
            q = tf.matmul(a_q, Q)
        return v, q
