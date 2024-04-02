import tensorflow as tf
import time

# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found. Please ensure TensorFlow GPU version is installed.")

# Generate random matrices
matrix_size = 1000
matrix_a = tf.random.normal([matrix_size, matrix_size])
matrix_b = tf.random.normal([matrix_size, matrix_size])

# Matrix multiplication on CPU
start_time = time.time()
result_cpu = tf.matmul(matrix_a, matrix_b)
end_time = time.time()
cpu_time = end_time - start_time
print("CPU time:", cpu_time)

# Matrix multiplication on GPU
with tf.device('/GPU:0'):
    start_time = time.time()
    result_gpu = tf.matmul(matrix_a, matrix_b)
    end_time = time.time()
    gpu_time = end_time - start_time
    print("GPU time:", gpu_time)

# Check if the results are the same
if tf.reduce_all(tf.equal(result_cpu, result_gpu)):
    print("Results are the same.")
else:
    print("Results are different.")

# Print speedup factor
if gpu_time > 0:
    speedup = cpu_time / gpu_time
    print("Speedup:", speedup)
else:
    print("Speedup cannot be calculated as GPU time is 0.")
