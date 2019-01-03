import tensorflow as tf
import pdb

pdb.set_trace()
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
print(cell.state_size)
inputs = tf.placeholder(tf.float32, shape=(32, 100))
h0 = cell.zero_state(32, tf.float32)
pdb.set_trace()
output, h1 = cell(inputs=inputs, state=h0)
print(h1)
print(h1.h, h1.h.shape)
print(h1.c, h1.c.shape)
#print(output, output.shape)
#test RNN
