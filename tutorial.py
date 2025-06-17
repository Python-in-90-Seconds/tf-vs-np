import numpy as np
import tensorflow as tf
import time

# 1. Basic Addition
a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])
c_np = a_np + b_np
print("NumPy ➤", c_np)

a_tf = tf.constant([1, 2, 3])
b_tf = tf.constant([4, 5, 6])
c_tf = a_tf + b_tf
print("TensorFlow ➤", c_tf.numpy())

# 2. Broadcasting
print("\n--- Broadcasting Example ---")
np_result = np.ones((3, 1)) + np.arange(3)
tf_result = tf.ones((3, 1)) + tf.cast(tf.range(3), dtype=tf.float32)
print("NumPy ➤\n", np_result)
print("TensorFlow ➤\n", tf_result.numpy())

# 3. Matrix Multiplication
A_np = np.random.rand(500, 500)
B_np = np.random.rand(500, 500)

A_tf = tf.constant(A_np)
B_tf = tf.constant(B_np)

# NumPy timing
start = time.time()
np.dot(A_np, B_np)
print("\nNumPy matmul time:", round(time.time() - start, 4), "sec")

# TensorFlow timing
start = time.time()
tf.matmul(A_tf, B_tf)
print("TensorFlow matmul time:", round(time.time() - start, 4), "sec")

# 4. Bonus: GPU Info
print("\nTensorFlow using GPU?" , tf.config.list_physical_devices('GPU') != [])
