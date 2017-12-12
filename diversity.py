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
    committed_current_gradient_norm_sum = tf.Variable(0.0, tf.float32)
    committed_diversity = tf.Variable(0.0, tf.float32)
 
    # Per batch
    filenames =[]
    for i in range(0, 10):
        filenames.append(file_prefix + "0" + str(i))
    for i in range(10, 22):
        filenames.append(file_prefix + str(i))

    train_filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=None, shuffle=True)
    train_reader = tf.TFRecordReader()
    _, example = train_reader.read(train_filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * FLAGS.batch_size
    first_batch = tf.train.shuffle_batch([example],          
            batch_size=16, capacity=capacity, 
            min_after_dequeue=min_after_dequeue)    
    example_batch = tf.train.shuffle_batch([example],          
            batch_size=FLAGS.batch_size, capacity=capacity, 
            min_after_dequeue=min_after_dequeue)    

    # Per example
    raw_example = tf.placeholder(tf.string)
    x_index, x_value, y = _parse_function(raw_example)

    w_gather = tf.gather(w, x_index)
    dot = tf.matmul(tf.transpose(tf.reshape(w_gather,[-1,1])),
            tf.reshape(x_value,[-1,1]))
    y_cast = tf.cast(y[0], tf.float32)
    sigmoid = y_cast *((y_cast * dot[0][0]) -1)
    candidate_gradient_values = tf.scalar_mul(sigmoid, x_value)
    
    x_index = tf.reshape(tf.cast(x_index, tf.int64), [-1,1])
    candidate_gradient = tf.SparseTensor(indices = x_index, 
            values = candidate_gradient_values, shape = [num_features])

    current_gradient_indices = tf.placeholder(tf.int64)
    current_gradient_values = tf.placeholder(tf.float32)
    current_gradient = tf.SparseTensor(indices = current_gradient_indices, 
            values = current_gradient_values, shape = [num_features])

    committed_gradient_indices = tf.placeholder(tf.int64)
    committed_gradient_values = tf.placeholder(tf.float32)
    committed_gradient = tf.SparseTensor(indices = committed_gradient_indices,
            values = committed_gradient_values, shape = [num_features])

    add = tf.sparse_add(current_gradient, candidate_gradient)
    add_norm = tf.reduce_sum(tf.square(add.values))
    denom = tf.add(committed_current_gradient_norm_sum, add_norm)

    num_before_norm = tf.sparse_add(add, committed_gradient)
    num = tf.reduce_sum(tf.square(num_before_norm.values))    
    better = tf.greater_equal(tf.div(denom, num), committed_diversity)
   
    #Per batch 
    local_gradient_indices = tf.placeholder(tf.int64)
    lo_ind = tf.reshape(local_gradient_indices, [-1])
    local_gradient_values = tf.placeholder(tf.float32)
    lo_val = tf.mul(local_gradient_values, eta)
    lo_val = tf.div(lo_val, FLAGS.batch_size * 1.0)
    update_weights = tf.scatter_sub(w, lo_ind, lo_val)   

    update_ccgns = committed_current_gradient_norm_sum.assign_add(
            tf.reduce_sum(tf.square(local_gradient_values)))

    global_gradient_indices = tf.placeholder(tf.int64)
    global_gradient_values = tf.placeholder(tf.float32)
    local_gradient = tf.SparseTensor(indices = local_gradient_indices,
            values = local_gradient_values, shape = [num_features])
    global_gradient = tf.SparseTensor(indices = global_gradient_indices,
            values = global_gradient_values, shape = [num_features])
    update_committed_gradient = tf.sparse_add(local_gradient, global_gradient)

    new_num = tf.reduce_sum(tf.square(update_committed_gradient.values))
    update_diversity = committed_diversity.assign(
            tf.div(committed_current_gradient_norm_sum, new_num)) 
 
    # Validation
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
        #print "Time\t" + str(datetime.datetime.now())
        tally = 0
        committed_i = [[0]]
        committed_v = [0]
        while tally < 2:
            l = sess.run(example_batch)
            current_i = [[0]]
            current_v = [0]
            for t in l:
                a = sess.run(add, 
                    {raw_example:t, 
                    current_gradient_indices:current_i, 
                    current_gradient_values:current_v,
                    committed_gradient_indices:committed_i,
                    committed_gradient_values:committed_v})
                current_i = a.indices
                current_v = a.values
            updates = [update_weights, update_ccgns, 
                    update_committed_gradient, update_diversity]
            _, _, u, _ = sess.run(updates, 
                    {local_gradient_indices:current_i, 
                    local_gradient_values:current_v,
                    global_gradient_indices:committed_i,
                    global_gradient_values:committed_v})
            committed_i = u.indices
            committed_v = u.values
            tally = tally + FLAGS.batch_size
            #print str(tally),
            #print str(datetime.datetime.now()),    
            #print committed_diversity.eval()

            if tally % 16384 == 0:
                print str(tally),
                print str(datetime.datetime.now()),    
                print committed_diversity.eval(),
                error_tally = 0
                for i in range(0, 10000):
                    error_tally +=sess.run(error)
                print str(error_tally)
                sys.stdout.flush()   

        print 'Done'
        coord.request_stop()
        coord.join(threads)
