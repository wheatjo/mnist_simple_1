import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference 

BATCH_SIZE = 100
LEARING_RATE_BASE = 0.8
LEARING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODE_SAVE_PATH = "F:/temsorflow1/google-MNist"
MODEL_NAME = "model.ckpt"

def train(mnist):
	x = tf.placeholder(
		tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
	y_ = tf.placeholder(
		tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y-input')
	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	y = mnist_inference.inference(x, regularizer)
	global_step = tf.Variable(0, trainable = False)

    #滑动平均
	variables_average = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_average_op = variables_average.apply(
		tf.trainable_variables())#返回一个需要训练的变量的列表，更新滑动平均变量
	cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits = y, labels = tf.argmax(y_, 1))
	cross_entroy_mean = tf.reduce_mean(cross_entroy)
	loss = cross_entroy_mean + tf.add_n(tf.get_collection('losses'))#将总损失数加起来
	#设置学习率，指数衰减学习率，具体看笔记
	learing_rate = tf.train.exponential_decay(
		LEARING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARING_RATE_DECAY
		)
	train_step = tf.train.GradientDescentOptimizer(learing_rate)\
	.minimize(loss, global_step = global_step)#传入global，自动更新参数global+1，所以学习率也相应更新

#确保train_step, variables_average_op都能按顺序运行
	with tf.control_dependencies([train_step, variables_average_op]):
		train_op = tf.no_op(name = 'train')#什么也不做，只是为了迎合需要的格式

	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss, global_step],
				                            feed_dict = {x: xs, y_:ys})#因为train_op什么也没做，所以用_打发
			if i % 1000 == 0:
				print("After %d training step(s), loss on trainging "
					"batch is %g." % (step, loss_value))
				saver.save(
					  sess, os.path.join(MODE_SAVE_PATH, MODEL_NAME), global_step = global_step)


def main(argv = None):
	mnist = input_data.read_data_sets("F:/temsorflow1/MNISt/MIN", one_hot = True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()