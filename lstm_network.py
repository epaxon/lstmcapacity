
import tensorflow as tf
import numpy as np


    
def init_weights(n_input, n_units):
    # Every ones knows this is how you do it of course
    return tf.truncated_normal([n_input, n_units], 0.0, 
                               tf.sqrt(2.0)/tf.sqrt(tf.cast(n_input + n_units, tf.float32)))




class LSTMnet(object):
    
    def __init__(self, hparams):

        self.n_input = hparams['n_inputs']
        self.n_units = hparams['n_units']
        
        self.forget_bias = hparams['forget_bias']
        
        self.w = tf.Variable(init_weights(self.n_input, 4*self.n_units)) # recurrent weights and gates
        self.v = tf.Variable(init_weights(self.n_units, self.n_input)) #output layer weights
        self.r = tf.Variable(init_weights(self.n_units, 4*self.n_units))
        self.b = tf.Variable(tf.zeros([4 * self.n_units]))
        self.bo = tf.Variable(tf.zeros([self.n_input])) #output bias
                
    
    def step(self, ch, x):
        """
        performs one lstm step
        @param ch: tuple of c and h state variables of lstm
        @param x: input vector
        """
        
        c,h = ch
        
        g = tf.matmul(x, self.w) + tf.matmul(h, self.r) + self.b
        
        # check out tf.split
        g_f = tf.sigmoid(g[:, 0:self.n_units] + self.forget_bias)
        g_i = tf.sigmoid(g[:, self.n_units:2*self.n_units])
        g_o = tf.sigmoid(g[:, 2*self.n_units:3*self.n_units])
        
        u   = tf.tanh(g[:, 3*self.n_units:4*self.n_units])
                
        new_c = g_f * c + g_i * u
        new_h = g_o * tf.tanh(new_c)
        new_y = tf.matmul(new_h, self.v) + self.bo
        
        return (new_c, new_h), new_y
    
    def get_new_state(self, batch_size):
        c = tf.Variable(tf.zeros((batch_size, self.n_units)), trainable=False)
        h = tf.Variable(tf.zeros((batch_size, self.n_units)), trainable=False)
        return c, h
    
    def reset_state_op(self, state):
        c,h = state
        return tf.group(c.assign(tf.zeros(c.get_shape())),
                        h.assign(tf.zeros(h.get_shape())))
    

    def store_state_op(self, state, storage):
        c,h = state
        cs, hs = storage
        return tf.group(cs.assign(c), hs.assign(h))

class ALSTMnet(object):
    
    def __init__(self, hparams):

        self.n_input = hparams['n_inputs']
        self.n_units = hparams['n_units']
        
        self.w = tf.Variable(init_weights(self.n_input, 6*self.n_units))
        self.r = tf.Variable(init_weights(self.n_units, 6*self.n_units))
        self.b = tf.Variable(np.zeros(6 * self.n_units))
                
    
    # Note: this acts by side effects on the state
    def step(self, h, x):
        g = tf.matmul(x, self.w) + tf.matmul(h, self.r) + self.b
        
        # check out tf.split
        g_f = tf.sigmoid(g[0:self.n_units])
        g_i = tf.sigmoid(g[self.n_units:2*self.n_units])
        g_o = tf.sigmoid(g[2*self.n_units:3*self.n_units])
        
        r_i = g[3*self.n_units:4*self.n_units]
        r_o = g[4*self.n_units:5*self.n_units]
        
        u   = tf.tanh(g[5*self.n_units:6*self.n_units])
        
        
        

        for i, layer in enumerate(self.layers):
            layer_state = state[i]
            with tf.name_scope('step'):
                forward_message = layer.step(layer_state, forward_message)

        return forward_message

