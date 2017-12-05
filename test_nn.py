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

FLAGS = None

def indexMax(list):
  """indexMax returns the index of the max element of the list."""
  return list.index(max(list))


def identifyOutputVelocity(sess, x, y_conv, keep_prob, input_data):
  """identifyDigitInImage apply the trained model to given image to identify the represented digit."""
  result = sess.run(y_conv, {x:[input_data], keep_prob: 1.0})[0].tolist()
  return indexMax(result)


def main():
  # Import data
  test_dataset, test_labelset = load_data(file_prefix="test")

  with tf.Session() as sess:

    # Restoring the trained model previously saved:
    saver = tf.train.import_meta_graph('./mnist_deep_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Trying to get back some required tensors variables from the restored graph:
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
    y_conv = graph.get_tensor_by_name("fc2/y_conv:0")

    # Now try to apply the model to randomly choosen test images, one by one:
    count = 0
    ok_count = 0
    while count < len(test_dataset):
      # Choosing a test image index:
      index = random.randint(0, len(test_dataset) - 1)
      test_data = test_dataset[index]

      # Applying the trained model to identify the digit from the test image:
      outputVelocity = identifyOutputVelocity(sess, x, y_conv, keep_prob, test_data)

      # Display the identified digit:
      print("The output velocity has been identified as a {}".format(outputVelocity))

      # Check the expected_digit from the test label of the choosen test image:
      expectedVelocity = indexMax(test_labelset[index])

      # Display the expected digit:
      print("Actually, the output velocity is a {}".format(expectedVelocity))

      # Count the correctly identified digits:
      if outputVelocity == expectedVelocity:
        ok_count += 1

      # Stop the loop after 10000 iterations
      count += 1

    # Display the measured accuracy during the test loop:
    print("Test accuracy = {}%".format(100 * (ok_count / count)))


if __name__ == '__main__':
  main()
