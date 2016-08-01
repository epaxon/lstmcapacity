import tensorflow as tf
import layers
import copy

class Network(object):
    def __init__(self, hparams):

        self.n_input = hparams['n_input']
        self.n_output = hparams['n_output']

        self.layers = []

        # create one or more rnn layers
        layer_inputs = self.n_input
        for i, n_unit in enumerate(hparams['layers']):
            with tf.name_scope('layer' + str(i)):
                layer =  layers.RLI_Layer(layer_inputs, n_unit, hparams)
            self.layers.append(layer)
            layer_inputs = n_unit


    # Note: this acts by side effects on the state
    def step(self, state, x, dt):
        forward_message = x

        for i, layer in enumerate(self.layers):
            layer_state = state[i]
            with tf.name_scope('step'):
                forward_message = layer.step(layer_state, forward_message, dt)

        return forward_message


    def get_new_state_store(self, n_state):
        state_store = []
        for layer in self.layers:
            layer_store = {}

            layer_store['h'] = tf.Variable(tf.zeros([n_state, layer.n_unit]),
                trainable=False, name='h')

            state_store.append(layer_store)

        return state_store

    def state_from_store(self, state_store):
        return [copy.copy(layer_store) for layer_store in state_store]

    # Records current state of the network in storage
    def store_state_op(self, state, state_store):
        store_ops = []
        for i in range(len(self.layers)):
            layer_state = state[i]
            layer_storage = state_store[i]

            for key in layer_state.keys():
                store_op = layer_storage[key].assign(layer_state[key])
                store_ops.append(store_op)

        return tf.group(*store_ops)


    # Resets the network state to zero
    def reset_state_op(self, state):
        reset_ops = []
        for i in range(len(self.layers)):
            layer_state = state[i]

            for key in layer_state.keys():
                state_var = layer_state[key]
                reset_op = state_var.assign(tf.zeros(state_var.get_shape()))
                reset_ops.append(reset_op)

        return tf.group(*reset_ops)
