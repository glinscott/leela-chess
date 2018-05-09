#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import random
import tensorflow as tf
import time
import bisect

NUM_STEP_TRAIN = 200
NUM_STEP_TEST = 2000
VERSION = 2

def weight_variable(name,shape):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.get_variable(name,
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.REGULARIZATION_LOSSES],
                              initializer=initial)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection
def bias_variable(name,shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name,initializer=initial)

# No point in learning bias weights as they are cancelled
# out by the BatchNorm layers's mean adjustment.
def bn_bias_variable(name,shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name,initializer=initial, trainable=False)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

# this function is copied from tensorflow mnist multigpu sample code
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = os.path.join(self.cfg['training']['path'], self.cfg['name'])

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model']['filters']
        self.RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks']

        # For exporting
        self.weights = []

        self.num_gpus = str(self.cfg['gpu']).count(',')+1
        self.multigpu = self.num_gpus > 1
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90, allow_growth=True, visible_device_list="{}".format(self.cfg['gpu']))
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.placeholder(tf.float32)

    def init(self, dataset, train_iterator, test_iterator):
        # TF variables
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, dataset.output_types, dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = self.session.run(train_iterator.string_handle())
        self.test_handle = self.session.run(test_iterator.string_handle())
        self.init_net(self.next_batch)

    def model(self,x,y,z):
        y_conv, z_conv = self.construct_net(x)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                    logits=y_conv)
        policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        mse_loss = \
            tf.reduce_mean(tf.squared_difference(z, z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = \
            tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']
        loss = pol_loss_w * policy_loss + val_loss_w *mse_loss + reg_term

        return y_conv, z_conv, policy_loss, mse_loss, reg_term, loss

    def parallel_model(self,next_batch):

        # run this stuff on the cpu.. dont want to move data back and forth between gpus
        # (unless have special hardware with fast interconnects between GPUs ?)
        with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
            self.x = next_batch[0]  # tf.placeholder(tf.float32, [None, 112, 8*8])
            self.y_ = next_batch[1] # tf.placeholder(tf.float32, [None, 1858])
            self.z_ = next_batch[2] # tf.placeholder(tf.float32, [None, 1])

            split_x = tf.split(self.x, self.num_gpus)
            split_y = tf.split(self.y_, self.num_gpus)
            split_z = tf.split(self.z_, self.num_gpus)

        device_grads = []
        policy_losses = []
        mse_losses = []
        reg_terms = []
        y_conv = []
        z_conv = []

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for i in range(self.num_gpus):
                device_x = split_x[i]
                device_y = split_y[i]
                device_z = split_z[i]
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                    # usualy called towers, but calling device here to avoid mixup with Residual towers
                    with tf.name_scope("device_%d" % i) as name_scope:

                        # reset the batch_norm and variable_name counters so the names becomes
                        # identical in all devices (towers)
                        self.batch_norm_count = 0
                        self.variable_count = 0

                        device_y_conv, device_z_conv = self.construct_net(device_x)

                        # Calculate loss on policy head
                        cross_entropy = \
                            tf.nn.softmax_cross_entropy_with_logits(labels=device_y,
                                                                    logits=device_y_conv)
                        device_policy_loss = tf.reduce_mean(cross_entropy)

                        # Loss on value head
                        device_mse_loss = \
                            tf.reduce_mean(tf.squared_difference(device_z, device_z_conv))

                        # Regularizer
                        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
                        #only get regularizer variables from gpu:0
                        if i == 0:
                            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

                        device_reg_term = \
                             tf.contrib.layers.apply_regularization(regularizer, reg_variables)

                        # save the reg_term for gpu 0 for tensorboard plotting
                        if i == 0:
                            reg_term = device_reg_term

                        # For training from a (smaller) dataset of strong players, you will
                        # want to reduce the factor in front of self.mse_loss here.
                        pol_loss_w = self.cfg['training']['policy_loss_weight']
                        val_loss_w = self.cfg['training']['value_loss_weight']
                        device_loss = pol_loss_w * device_policy_loss  + val_loss_w * device_mse_loss + device_reg_term

                        # compute the gradients for the current device
                        grad = self.opt_op.compute_gradients(device_loss,colocate_gradients_with_ops=True)

                        # let the training device gather all the data from the devices
                        with tf.device(self.train_device):
                            y_conv.append(device_y_conv)
                            z_conv.append(device_z_conv)
                            policy_losses.append(device_policy_loss)
                            mse_losses.append(device_mse_loss)
                            device_grads.append(grad)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all devices(towers). This is all done on the training device
        grads = average_gradients(device_grads)

        return tf.concat(y_conv, axis=0), \
                tf.concat(z_conv, axis=0), \
                tf.add_n(policy_losses)/self.num_gpus, \
                tf.add_n(mse_losses)/self.num_gpus, \
                reg_term, \
                grads;

    def init_net(self, next_batch):

        self.batch_norm_count = 0
        self.variable_count = 0
        # used to make sure we only a weight once to weights list
        self.weights_added = False

        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.cfg['training']['lr_values'].sort(reverse=True)
        self.lr = self.cfg['training']['lr_values'][0]

        if not self.multigpu:
            self.x = next_batch[0]  # tf.placeholder(tf.float32, [None, 112, 8*8])
            self.y_ = next_batch[1] # tf.placeholder(tf.float32, [None, 1858])
            self.z_ = next_batch[2] # tf.placeholder(tf.float32, [None, 1])

            self.y_conv, self.z_conv, self.policy_loss, self.mse_loss, self.reg_term, loss = \
                self.model(x=self.x, y=self.y_, z=self.z_)

            # You need to change the learning rate here if you are training
            # from a self-play training set, for example start with 0.005 instead.
            opt_op = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(self.update_ops):
                self.train_op = \
                    opt_op.minimize(loss, global_step=self.global_step,colocate_gradients_with_ops=self.multigpu)

            correct_prediction = \
                tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
        else:
            self.train_device = self.cfg['training']['multi_gpu_train_device'] if 'multi_gpu_train_device' in self.cfg['training'] else '/cpu:0'
            with tf.device(self.train_device):
                # You need to change the learning rate here if you are training
                # from a self-play training set, for example start with 0.005 instead.
                self.opt_op = tf.train.MomentumOptimizer(
                    learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

                self.y_conv, self.z_conv, self.policy_loss, self.mse_loss, self.reg_term, gradient = \
                    self.parallel_model(next_batch)

                # Apply the gradients to adjust the shared variables.
                apply_gradient_op = self.opt_op.apply_gradients(gradient, global_step=self.global_step)

                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(self.update_ops):
                    self.train_op = tf.group(apply_gradient_op)
                correct_prediction = \
                    tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
                self.accuracy = tf.reduce_mean(correct_prediction)

        self.avg_policy_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/{}-test".format(self.cfg['name'])), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/{}-train".format(self.cfg['name'])), self.session.graph)

        self.init = tf.global_variables_initializer()
        # change to using relative paths so can easilly move the saved nets
        self.saver = tf.train.Saver(save_relative_paths=True)

        self.session.run(self.init)

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            # Keyed batchnorm weights
            if isinstance(weights, str):
                work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                new_weight = tf.constant(new_weights[e])
                self.session.run(tf.assign(work_weights, new_weight))
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [2, 3, 1, 0])))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [1, 0])))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.session.run(weights.assign(new_weight))
        #This should result in identical file to the starting one
        #self.save_leelaz_weights('restored.txt')

    def restore(self, file):
        # have to append the full path since using relative paths
        file = self.root_dir+'/'+file
        print("Restoring from {0}".format(file))

        # clear devices from meta graph so can reconfigure gpu and trainingg device settings
        # without recreating graph from weights file
        new_saver = tf.train.import_meta_graph(file+'.meta',
                                               clear_devices=True)
        new_saver.restore(self.session, file)

    def process(self, batch_size, test_batches):
        if not self.time_start:
            self.time_start = time.time()

        # Run training for this batch
        policy_loss, mse_loss, reg_term, _, _ = self.session.run(
            [self.policy_loss, self.mse_loss, self.reg_term, self.train_op,
                self.next_batch],
            feed_dict={self.training: True, self.learning_rate: self.lr, self.handle: self.train_handle})

        steps = tf.train.global_step(self.session, self.global_step)

        # Determine learning rate
        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = (steps-1) % self.cfg['training']['total_steps']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]

        # Keep running averages
        # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
        # get comparable values.
        mse_loss /= 4.0
        self.avg_policy_loss.append(policy_loss)
        self.avg_mse_loss.append(mse_loss)
        self.avg_reg_term.append(reg_term)
        if steps % NUM_STEP_TRAIN == 0:
            pol_loss_w = self.cfg['training']['policy_loss_weight']
            val_loss_w = self.cfg['training']['value_loss_weight']
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = batch_size * (NUM_STEP_TRAIN / elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print("step {}, lr={:g} policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                steps, self.lr, avg_policy_loss, avg_mse_loss, avg_reg_term,
                # Scale mse_loss back to the original to reflect the actual
                # value being optimized.
                # If you changed the factor in the loss formula above, you need
                # to change it here as well for correct outputs.
                pol_loss_w * avg_policy_loss + val_loss_w * avg_mse_loss + avg_reg_term,
                speed))
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
                tf.Summary.Value(tag="Reg term", simple_value=avg_reg_term),
                tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
            self.train_writer.add_summary(train_summaries, steps)
            self.time_start = time_end
            self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []

        if steps % NUM_STEP_TEST == 0:
            sum_accuracy = 0
            sum_mse = 0
            sum_policy = 0
            for _ in range(0, test_batches):
                test_policy, test_accuracy, test_mse, _ = self.session.run(
                    [self.policy_loss, self.accuracy, self.mse_loss,
                     self.next_batch],
                    feed_dict={self.training: False,
                               self.handle: self.test_handle})
                sum_accuracy += test_accuracy
                sum_mse += test_mse
                sum_policy += test_policy
            sum_accuracy /= test_batches
            sum_accuracy *= 100
            sum_policy /= test_batches
            # Additionally rescale to [0, 1] so divide by 4
            sum_mse /= (4.0 * test_batches)
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
            self.test_writer.add_summary(test_summaries, steps)
            print("step {}, policy={:g} training accuracy={:g}%, mse={:g}".\
                format(steps, sum_policy, sum_accuracy, sum_mse))

        if steps % self.cfg['training']['total_steps'] == 0:
            path = os.path.join(self.root_dir, self.cfg['name'])
            save_path = self.saver.save(self.session, path, global_step=steps)
            print("Model saved in file: {}".format(save_path))
            leela_path = path + "-" + str(steps) + ".txt"
            self.save_leelaz_weights(leela_path) 
            print("Weights saved in file: {}".format(leela_path))

    def save_leelaz_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write("{}".format(VERSION))
            for weights in self.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                # Keyed batchnorm weights
                if isinstance(weights, str):
                    work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def get_variable_name(self):
        result = "variable_" + str(self.variable_count)
        self.variable_count += 1
        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = weight_variable(self.get_variable_name(),[filter_size, filter_size,
                                  input_channels, output_channels])
        b_conv = bn_bias_variable(self.get_variable_name(),[output_channels])

        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        weight_key = self.get_batchnorm_key()
        if not self.weights_added:
            self.weights.append(W_conv)
            self.weights.append(b_conv)
            self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
            self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key, reuse=tf.AUTO_REUSE):
            h_bn = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        orig = tf.identity(inputs)
        W_conv_1 = weight_variable(self.get_variable_name(),[3, 3, channels, channels])
        b_conv_1 = bn_bias_variable(self.get_variable_name(),[channels])
        weight_key_1 = self.get_batchnorm_key()

        if not self.weights_added:
            self.weights.append(W_conv_1)
            self.weights.append(b_conv_1)
            self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
            self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

        # Second convnet
        W_conv_2 = weight_variable(self.get_variable_name(),[3, 3, channels, channels])
        b_conv_2 = bn_bias_variable(self.get_variable_name(),[channels])
        weight_key_2 = self.get_batchnorm_key()

        if not self.weights_added:
            self.weights.append(W_conv_2)
            self.weights.append(b_conv_2)
            self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
            self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key_1, reuse=tf.AUTO_REUSE):
            h_bn1 = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv_1),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2, reuse=tf.AUTO_REUSE):
            h_bn2 = \
                tf.layers.batch_normalization(
                    conv2d(h_out_1, W_conv_2),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))
        return h_out_2

    def construct_net(self, planes):
        # NCHW format
        # batch, 112 input channels, 8 x 8
        x_planes = tf.reshape(planes, [-1, 112, 8, 8])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=112,
                               output_channels=self.RESIDUAL_FILTERS)
        # Residual tower
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, self.RESIDUAL_FILTERS)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=32)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 32*8*8])
        W_fc1 = weight_variable(self.get_variable_name(),[32*8*8, 1858])
        b_fc1 = bias_variable(self.get_variable_name(),[1858],)
        if not self.weights_added:
            self.weights.append(W_fc1)
            self.weights.append(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1, name='policy_head')

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=32)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 32*8*8])
        W_fc2 = weight_variable(self.get_variable_name(),[32 * 8 * 8, 128])
        b_fc2 = bias_variable(self.get_variable_name(),[128])
        if not self.weights_added:
            self.weights.append(W_fc2)
            self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable(self.get_variable_name(),[128, 1])
        b_fc3 = bias_variable(self.get_variable_name(),[1])
        if not self.weights_added:
            self.weights.append(W_fc3)
            self.weights.append(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3), name='value_head')

        self.weights_added = True
        return h_fc1, h_fc3
