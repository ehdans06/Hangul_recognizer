import tensorflow as tf
from load_data import load_data

max_epoch = 200
batch_size = 100

data = load_data('euc-kr')

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y1_conv = tf.layers.conv2d(inputs=x, filters=6, kernel_size=5, padding='same', activation=tf.nn.relu)    # kernel_initializer=tf.contrib.layers.xavier_initializer
y1_pool = tf.layers.max_pooling2d(inputs=y1_conv, pool_size=2, strides=1, padding='same')
y2_conv = tf.layers.conv2d(inputs=y1_pool, filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)    # tf.contrib.layers.xavier_initializer
y2_pool = tf.layers.max_pooling2d(inputs=y2_conv, pool_size=2, strides=1, padding='same')
y2_flat = tf.contrib.layers.flatten(inputs=y2_conv)
y3 = tf.layers.dense(inputs=y2_flat, units=120, activation=tf.nn.relu)    # tf.contrib.layers.xavier_initializer
y4 = tf.layers.dense(inputs=y3, units=84, activation=tf.nn.relu)    # tf.contrib.layers.xavier_initializer
y5 = tf.layers.dense(inputs=y4, units=19, activation=tf.nn.relu)    # tf.contrib.layers.xavier_initializer
y_label = tf.placeholder(dtype=tf.float32, shape=[None, 19])

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y5))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(y5, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
print('=====Training=====')
for it in range(max_epoch):
    print("Iteration %d/%d" % (it + 1, max_epoch))
    batch_xs, batch_ys = data.train.next_batch(batch_size)
    _, cost_val, acc_val = sess.run([train_step, loss, accuracy], feed_dict={x: batch_xs, y_label: batch_ys})
    print('Mean Cost : %f' % cost_val)
    print('Train Accuracy : %f' % acc_val)

# Test
# print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))