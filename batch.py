import tensorflow as tf
import os
import sys
import datetime
import math
tf.app.flags.DEFINE_integer("batch_size", 32, "Size of batch")
FLAGS = tf.app.flags.FLAGS

num_features = 33762578
file_prefix = "/home/ubuntu/tfscripts/data/tfrecords"
eta = 0.01
def _parse_function(example_proto):
    features = {
        'label': tf.FixedLenFeature([1], dtype=tf.int64),
        'index': tf.VarLenFeature(dtype=tf.int64),
        'value': tf.VarLenFeature(dtype=tf.float32),
    }
    example = tf.parse_single_example(example_proto, features)
    y = example['label']
    x_index = example['index']
    x_value = example['value']
    x_index = tf.sparse_tensor_to_dense(x_index)
    x_value = tf.sparse_tensor_to_dense(x_value)
    return x_index, x_value, y

g = tf.Graph()

with g.as_default():

    w = tf.Variable(tf.random_uniform([num_features]), name="model")
    
    filenames =[]
    for i in range(0, 10):
        filenames.append(file_prefix + "0" + str(i))
    for i in range(10, 22):
        filenames.append(file_prefix + str(i))

    train_filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=None, shuffle=True)
    train_reader = tf.TFRecordReader()
    _, example = train_reader.read(train_filename_queue)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * FLAGS.batch_size
    example_batch = tf.train.shuffle_batch([example],          
            batch_size=FLAGS.batch_size, capacity=capacity, 
            min_after_dequeue=min_after_dequeue)    

    raw_example = tf.placeholder(tf.string)
    x_index, x_value, y = _parse_function(raw_example)

    w_gather = tf.gather(w, x_index)
    dot = tf.matmul(tf.transpose(tf.reshape(w_gather,[-1,1])),
            tf.reshape(x_value,[-1,1]))
    y_cast = tf.cast(y[0], tf.float32)
    sigmoid = y_cast *((y_cast * dot[0][0]) -1)
    new_gradient = tf.scalar_mul(sigmoid, x_value)
    #new_gradient = tf.mul(new_gradient, eta)

    x_index = tf.reshape(tf.cast(x_index, tf.int64), [-1,1])
    new_gradient_sparse = tf.SparseTensor(indices = x_index, 
            values = new_gradient, shape = [num_features])

    gradient_indices = tf.placeholder(tf.int64)
    gradient_values = tf.placeholder(tf.float32)
    gradient_sparse = tf.SparseTensor(indices = gradient_indices, 
            values = gradient_values, shape = [num_features])

    add = tf.sparse_add(gradient_sparse, new_gradient_sparse)
     
    #eta = tf.placeholder(tf.float32)
    local_gradient_indices = tf.placeholder(tf.int32)
    lo = tf.reshape(local_gradient_indices, [-1])
    local_gradient = tf.placeholder(tf.float32)
    lo_val = tf.mul(local_gradient, eta)
    lo_val = tf.div(lo_val, 
            FLAGS.batch_size * 1.0)
    update_op = tf.scatter_sub(w, lo, lo_val)   

    testfiles = []
    testfiles.append(file_prefix + str(22))
    test_filename_queue = tf.train.string_input_producer(
            testfiles, num_epochs=None, shuffle=True)
    test_reader = tf.TFRecordReader()
    _, test = test_reader.read(test_filename_queue)
    xt_index, xt_value, yt = _parse_function(test)
    wt_gather = tf.gather(w, xt_index)
    yp = tf.matmul(tf.transpose(tf.reshape(wt_gather, [-1,1])),
            tf.reshape(xt_value,[-1,1]))
    match = tf.equal(yt[0], tf.to_int64(tf.sign(yp[0][0])))
    error = tf.cond(match, lambda:tf.constant(0), lambda:tf.constant(1))
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)   
        print "Time\t" + str(datetime.datetime.now())
        tally = 0
        #lr = 0.01
        while tally < 100000000:
            l = sess.run(example_batch)
            g_i = [[0]]
            g_v = [0]
            for t in l:
                g = sess.run(add, 
                    {raw_example:t, gradient_indices:g_i, gradient_values:g_v})
                g_i = g.indices
                g_v = g.values

            a=sess.run(update_op, {#eta: lr,
                    local_gradient_indices: g_i, local_gradient: g_v})
            tally = tally + FLAGS.batch_size
            #lr = lr*math.exp(-0.01*tally/FLAGS.batch_size) + 0.00001
            if tally % 16384 == 0:
                print "Count\t" + str(tally)
                print "Time\t" + str(datetime.datetime.now())
                
                error_tally = 0
                for i in range(0, 10000):
                    error_tally +=sess.run(error)
                print "Error\t" + str(error_tally)
                sys.stdout.flush()   

        print 'Done'
        coord.request_stop()
        coord.join(threads)
