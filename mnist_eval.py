import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 100
def evaluate(mnist):
	with tf.Graph().as_default() as g:#形成一张图，在此图运行
		x = tf.placeholder(
			tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
		y_ = tf.placeholder(
			tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y-input')
		validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

		y = mnist_inference.inference(x, None)

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))#tf.argmax(y, 1)得到一个向量，equal得到一个True，False的向量
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#cast类型转换函数 如int32到float32


		variable_averages = tf.train.ExponentialMovingAverage(
			mnist_train.MOVING_AVERAGE_DECAY)#衰减率 控制模型更新速度
		variables_to_restore = variable_averages.variables_to_restore()#加载滑动平均量（影子变量
		saver = tf.train.Saver(variables_to_restore)#指定保存变量variables_to_restore

		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(
					mnist_train.MODE_SAVE_PATH)#加载模型
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					#通过文件名得到模型保存时迭代的轮数 
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]#分割字符串， 文件路径是一个字符串 有/和-
					accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
					print("After %s teaining step(s), validation"
						  "accuracy = %g" % (global_step, accuracy_score))
				else:
					print('No checkpoint file found')
					return 
			time.sleep(EVAL_INTERVAL_SECS)

def main(argv = None):
	mnist = input_data.read_data_sets("F:/temsorflow1/MNISt/MIN", one_hot = True)
	evaluate(mnist)

if __name__ == '__main__':
	tf.app.run()