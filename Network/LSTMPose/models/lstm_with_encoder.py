import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1
import math

# TF2 compatibility: OutputProjectionWrapper replacement
class OutputProjectionWrapper(tf.nn.rnn_cell.RNNCell):
    """Wrapper that projects RNN output to a specified size."""
    def __init__(self, cell, output_size, activation=None):
        super(OutputProjectionWrapper, self).__init__()
        self._cell = cell
        self._output_size = output_size
        self._activation = activation
        self._kernel = None
        self._bias = None

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        output, new_state = self._cell(inputs, state)
        # Use tf.get_variable to match original checkpoint variable names
        if self._kernel is None:
            input_size = output.get_shape().as_list()[-1]
            self._kernel = tf.get_variable('kernel', [input_size, self._output_size])
            self._bias = tf.get_variable('bias', [self._output_size], initializer=tf.zeros_initializer())
        projected = tf.matmul(output, self._kernel) + self._bias
        if self._activation is not None:
            projected = self._activation(projected)
        return projected, new_state


def pred_net(features,
             confidence,
             seq_lens=None,
             is_training=False,
             is_inference=False,
             is_initial=tf.constant(False, tf.bool, []),
             last_state=None,
             lstm_num_layer=3,
             lstm_num_unit=256):
    """
    Direct regression.
    :param features: angles, tf.float32 [batch_size, num_seq, num_feat], (-pi, pi)
    :param confidence: confidence coefficient, tf.float32 [batch_size, num_seq, num_feat] [0, 1]
    :param is_training:
    :return: output. (-pi, pi)
    """
    assert not (is_inference and is_training)
    assert not (is_inference and last_state is None)
    assert not (not is_inference and seq_lens is None)

    num_feat = features.get_shape().as_list()[-1]

    with tf.variable_scope('pred_net'):
        features = features / math.pi

        if confidence is not None:
            # inputs = tf.concat([features, confidence], -1)
            a_s = tf.layers.Dense(64)
            a_t = tf.layers.Dense(64)
            a_a = tf.layers.Dense(128)
            if not is_inference:
                inputs = tf.concat([tf.map_fn(lambda i: a_s.apply(i), features),
                                   tf.map_fn(lambda i: a_t.apply(i), confidence)], -1)
                inputs = tf.map_fn(lambda i: a_a.apply(i), inputs)
            else:
                inputs = a_a.apply(tf.concat([a_s.apply(features), a_t.apply(confidence)], -1))
        else:
            a_s = tf.layers.Dense(64)
            if not is_inference:
                inputs = tf.map_fn(lambda i: a_s.apply(i), features)
            else:
                inputs = a_s.apply(features)

        if lstm_num_layer == 1:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_num_unit)
        else:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(lstm_num_unit) for _ in range(lstm_num_layer)])
        lstm_cell = OutputProjectionWrapper(lstm_cell, num_feat, activation=None)

        if not is_inference:
            # inputs = tf.unstack(inputs, axis=1)
            # outputs, _ = tf.nn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
            # outputs = tf.stack(outputs, 1)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, seq_lens, dtype=tf.float32, parallel_iterations=128)
            outputs = outputs * math.pi
            outputs = tf.clip_by_value(outputs, -math.pi, math.pi)
            return outputs
        else:
            state = tf.cond(is_initial, lambda: lstm_cell.zero_state(1, tf.float32), lambda: last_state)
            outputs, state = lstm_cell.call(inputs, state)
            outputs = outputs * math.pi
            outputs = tf.clip_by_value(outputs, -math.pi, math.pi)
            return outputs, state


if __name__ == '__main__':
    fake_features = tf.random_normal([1, 25])
    fake_conf = tf.ones([1, 25])
    fake_label = tf.random_normal([1, 25])

    default_state = []
    for _ in range(3):
        default_state.append(tf.nn.rnn_cell.LSTMStateTuple(tf.constant(0, tf.float32, [1, 256]),
                                                           tf.constant(0, tf.float32, [1, 256])))
    fake_outputs = pred_net(fake_features, fake_conf,
                            is_inference=True,
                            last_state=tuple(default_state))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out_val = sess.run(fake_outputs)
        pass
