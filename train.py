import tensorflow as tf
from load_data import load_data

max_epoch = 300
batch_size = 100
num_cho, num_jung, num_jong = 19, 21, 28

data = load_data('euc-kr')

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y1_conv = tf.layers.conv2d(inputs=x, filters=20, kernel_size=5, padding='same', activation=tf.nn.relu)
y1_pool = tf.layers.max_pooling2d(inputs=y1_conv, pool_size=2, strides=2, padding='same')
y2_conv = tf.layers.conv2d(inputs=y1_pool, filters=50, kernel_size=5, padding='same', activation=tf.nn.relu)
y2_pool = tf.layers.max_pooling2d(inputs=y2_conv, pool_size=2, strides=2, padding='same')
y2_flat = tf.contrib.layers.flatten(inputs=y2_conv)
y3 = tf.layers.dense(inputs=y2_flat, units=500, activation=tf.nn.relu)
y4 = tf.layers.dense(inputs=y3, units=68, activation=tf.nn.relu)
y_hat = tf.nn.softmax(y4)
y_label = tf.placeholder(dtype=tf.float32, shape=[None, 68])

y_hat_cho = tf.slice(y_hat, [0, 0], [-1, num_cho])
y_hat_jung = tf.slice(y_hat, [0, num_cho], [-1, num_jung])
y_hat_jong = tf.slice(y_hat, [0, num_cho + num_jung], [-1, -1])
y_label_cho = tf.slice(y_label, [0, 0], [-1, num_cho])
y_label_jung = tf.slice(y_label, [0, num_cho], [-1, num_jung])
y_label_jong = tf.slice(y_label, [0, num_cho + num_jung], [-1, -1])

# Define loss and optimizer
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_hat))
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y_hat), axis=[1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

# Define accuracy
accuracy = []
accuracy.append(tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(y_hat_cho, 1), tf.argmax(y_label_cho, 1)), tf.float32
)))
accuracy.append(tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(y_hat_jung, 1), tf.argmax(y_label_jung, 1)), tf.float32
)))
accuracy.append(tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(y_hat_jong, 1), tf.argmax(y_label_jong, 1)), tf.float32
)))

# tensorboard summary
cost_summary = tf.summary.scalar('cost', loss)
accuracy_summary_1 = tf.summary.scalar('accuracy_cho', accuracy[0])
accuracy_summary_2 = tf.summary.scalar('accuracy_jung', accuracy[1])
accuracy_summary_3 = tf.summary.scalar('accuracy_jong', accuracy[2])
merged = tf.summary.merge_all()

# Start session
sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('/home/ms/sketch/JetBrains/Pycharm/DLproj/tensorboard/train', sess.graph)
tf.global_variables_initializer().run()

# Train
print('=====Training=====')
for it in range(max_epoch):
    print("Iteration %d/%d" % (it + 1, max_epoch))
    batch_xs, batch_ys = data.train.next_batch(batch_size)
    result = sess.run([train_step, loss] + accuracy, feed_dict={x: batch_xs, y_label: batch_ys})
    print('Mean Cost : %f' % result[1])
    print('Train Accuracy : %f, %f, %f' % (result[2], result[3], result[4]))
    if it % 10 == 0:
        train_writer.add_summary(merged.eval(feed_dict={x: batch_xs, y_label: batch_ys}), it)

# Test
# print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))