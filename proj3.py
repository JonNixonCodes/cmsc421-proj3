# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import random

import tensorflow as tf

from generate_data import *
import maketrack

FLAGS = None


def indexMax(list):
	"""indexMax returns the index of the max element of the list."""
	return list.index(max(list))


def identifyOutputVelocity(sess, x, y_conv, keep_prob, input_data):
	"""identifyDigitInImage apply the trained model to given image to identify the represented digit."""
	result = sess.run(y_conv, {x:[input_data], keep_prob: 1.0})[0].tolist()
	return indexMax(result)


def main(state, f_line, walls):
	input_vector = transform_data(state, f_line, walls)

	with tf.Session() as sess:

		# Restoring the trained model previously saved:
		saver = tf.train.import_meta_graph('./mnist_deep_model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))

		# Trying to get back some required tensors variables from the restored graph:
		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name("x:0")
		keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
		y_conv = graph.get_tensor_by_name("fc2/y_conv:0")

		# Now apply the model input vector:

		# Applying the trained model to identify the output change in velocity from input:
		# output is in index format
		dVelIndex = identifyOutputVelocity(sess, x, y_conv, keep_prob, input_vector)

	#convert change in velocity to velocity
	dU, dV = convertOutputIndexToChangeInVelocity(dVelIndex)

	# Display the identified digit:
	print("The output velocity has been identified as a {}".format(outputVelocity))

	(u,v) = state[1]
	return ((u+dU),(v+dV))

def test():
	problem = maketrack.main()
	state = (problem[0], (0,0))    # initial state
	f_line = problem[1]
	walls = problem[2]
	(u,v) = main(state,f_line,walls)
