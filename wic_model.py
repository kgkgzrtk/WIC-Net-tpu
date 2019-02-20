from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from PIL import Image


def _leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
    return tf.layers.batch_normalization(
            x, momentum=0.9, epsilon=1e-5, training=is_training, fused=True, name=name)


def _spec_norm(w):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_ = tf.matmul(u_hat, tf.transpose(w))
    v_hat = tf.nn.l2_normalize(v_)
    u_ = tf.matmul(v_hat, w)
    u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w/sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm
    

def _dense(x, channels, name):
    return tf.layers.dense(
            x, channels,
            kernel_initializer=tf.glorot_uniform_initializer(stddev=0.02),
            use_bias=False,
            name=name)


def _conv2d(x, out_dim, c, k, name, sn=False, use_bias=False, padding='SAME'):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('w', [c, c, x.get_shape().dims[-1].value, out_dim], initializer=tf.glorot_uniform_initializer(stddev=0.02))
        if sn:
            W = _spec_norm(W)
        y = tf.nn.conv2d(x, W, strides=[1, k, k, 1], padding=padding) 
        if use_bias:
            b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
            return tf.add(y,b)
        else: return y


def _bilinear(x, out_shape):
    b, h, w, ch = x.shape.as_list()
    s = 2
    size = (2 * s - s % 2)
    karnel = np.zeros([size, size], dtype=np.float32)
    scale_factor = (size + 1)//2
    if size % 2 == 1:
        c = scale_factor - 1
    else:
        c = scale_factor - 0.5
    for i in range(size):
        for j in range(size):
            karnel[i,j] = (1 - abs(i - c)/scale_factor)*(1 - abs(j - c)/scale_factor)
    w_filter = np.zeros([size, size, ch, ch])
    for i in range(ch):
        w_filter[:,:,i,i] = karnel
    conv_filter = tf.constant(w_filter, dtype=tf.float32)
    return tf.nn.conv2d_transpose(x, conv_filter, out_shape, strides=[1,s,s,1], padding='SAME')



def _deconv2d(x, out_dim, c, k, name, use_bias=False):
    with tf.variable_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        out_shape = [x_shape[0], x_shape[1]*k, x_shape[2]*k, out_dim]
        W = tf.get_variable('w', [c, c, out_dim, x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        W_ = _spec_norm(W)
        y = tf.nn.conv2d_transpose(x, W_, out_shape, strides=[1, k, k, 1], padding='SAME')
        if use_bias:
            b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
            return y+b
        else: return y


def _pixel_shuffler(image, out_shape, r=2, c=4, name='ps'):
    with tf.variable_scope(name) as scope:
        y_conv = _conv2d(image, out_shape[-1]*(r**2), c=c, k=1, name='first_conv')
        y_list = tf.split(y_conv, out_shape[3], 3)
        pix_map_list = []
        for y in y_list:
            b, h, w, c = y.get_shape().as_list()
            pix_map = tf.reshape(y, [b, h, w, r, r])
            pix_map = tf.transpose(pix_map, perm=[0, 1, 2, 4, 3])
            pix_map = tf.split(pix_map,h,1)
            pix_map = tf.concat([tf.squeeze(m,1) for m in pix_map],2)
            pix_map = tf.split(pix_map,w,1)
            pix_map = tf.concat([tf.squeeze(m,1) for m in pix_map],2)
            pix_map = tf.reshape(pix_map, [b, h*r, w*r, 1])
            pix_map_list.append(pix_map)
        out = tf.concat(pix_map_list, 3)
        return out


def _upsampling(x, name, mode='bi'):
    b, h, w, c = x.get_shape().as_list()
    out_shape = [b, h*2, w*2, c]
    if mode == 'deconv':
        return _deconv2d(x, x.get_shape().dims[-1].value, 3, 2, name=name) 
    elif mode == 'bi':
        # BC Edge bug
        return tf.image.resize_bilinear(x, [x.shape[1]*2, x.shape[2]*2], align_corners=True, name=name)
        #return _bilinear(x, out_shape)

def _downsampling(x, name):
    return tf.layers.average_pooling2d(x, 2, 2, padding='same', name=name)


def embedding(y, in_size, out_size, scope):
    with tf.variable_scope(scope):
        V = tf.get_variable('w', [in_size, out_size], initializer=tf.glorot_uniform_initializer())
        V_ = _spec_norm(V)
        o = tf.matmul(y, V_)
    return o


def _res_block_enc(x, out_dim, is_training, scope='res_enc'):
    with tf.variable_scope(scope):
        c_s = _downsampling(x, name='s_down')
        c_s = _conv2d(c_s, out_dim, 1, 1, name='s_c')
        x = tf.nn.relu(_bach_norm(x, is_training, name='bn1'))
        x = _downsampling(x, name='down')
        x = _conv2d(x, out_dim, 3, 1, name='c1')
        x = tf.nn.relu(_bach_norm(x, is_training, name='bn2'))
        x = _conv2d(x, out_dim, 3, 1, name='c2')
        x = tf.add(x, c_x)
        return x


def _res_block_down(x, out_dim, is_training, scope='res_down'):
    with tf.variable_scope(scope):
        c_s = _conv2d(x, out_dim, 1, 1, name='s_c')
        c_s = _downsampling(c_s, name='s_down')
        x = _conv2d(tf.nn.relu(x), out_dim, 3, 1, name='c1')
        x = _conv2d(tf.nn.relu(x), out_dim, 3, 1, name='c2')
        x = _downsampling(x, name='down')
        x = tf.add(x, c_s)
        return x


def _res_block_up(x, out_dim, is_training, scope='res_up'):
    with tf.variable_scope(scope):
        c_s = _upsampling(x, name='s_up', mode='bi')
        c_s = _conv2d(c_s, out_dim, 1, 1, name='s_c' padding='VALID')
        #x = tf.layers.dropout(x, rate=0.3, training=is_training)
        x = _leaky_relu(_batch_norm(x, is_training, name='bn1'))
        x = _upsampling(x, name='up', mode='bi')
        x = _conv2d(x, out_dim, 3, 1, name='c1')
        x = _leaky_relu(_batch_norm(x, is_training, name='bn2'))
        x = _conv2d(x, out_dim, 3, 1, name='c2')
        x = tf.add(x, c_s)
        return x


def discriminator(x, a, is_training=True, scope='Discriminator'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dis_dim = 64
        feat_li = []
        for i in range(5):
            if i<4: x = tf.layers.dropout(x, rate=0.5, training=is_training)
            x = _res_block_down(x, dis_dim*(2**i), is_training, scope='b_down_'+str(i))
            feat_li.append(x)
        x_feat = _leaky_relu(x)
        x = tf.reduce_sum(x_feat, axis=[1, 2])
        emb_a = embedding(a, 6, x.shape[-1], scope='emb')
        emb = tf.reduce_sum(emb_a * x, axis=1, keepdims=True)
        o = emb + _dense(x, 1, name='fc')
        return o, feat_li

def generator(x, is_training=True, scope='Generator'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        ch = 64
        x = _dense(x, 4*4*ch, name='fc')
        x = tf.reshape(x, [-1, 4, 4, ch])
        for i in range(5):
            x = _res_block_up(x, ch, is_training, scope='b_up_'+str(i))
            ch = ch//2
        x = _leaky_relu(_batch_norm(x, is_training, name='bn'))
        x = _conv2d(x, 3, 3, 1, name='final_c')
        x = tf.tanh(x)
        return x
