import tensorflow as tf
from load_data import load_data

max_epoch = 300
batch_size = 100
num_cho, num_jung, num_jong = 19, 21, 28

data = load_data('euc-kr')

#common
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y_label = tf.placeholder(dtype=tf.float32, shape=[None, 68])

#for cho
y11_conv = tf.layers.conv2d(inputs=x, filters=10, kernel_size=5, padding='same')
bn11 = tf.layers.batch_normalization(inputs=y11_conv, axis=-1, momentum=0.99, epsilon=0.001)
y12_conv = tf.layers.conv2d(inputs=bn11, filters=25, kernel_size=5, padding='same')
bn12 = tf.layers.batch_normalization(inputs=y12_conv, axis=-1, momentum=0.99, epsilon=0.001)
y12_flat = tf.contrib.layers.flatten(inputs=bn12)
y13 = tf.layers.dense(inputs=y12_flat, units=250, activation=tf.nn.relu)
y14 = tf.layers.dense(inputs=y13, units=num_cho)
y_hat_cho = tf.nn.softmax(y14)


#for jung
y21_conv = tf.layers.conv2d(inputs=x, filters=10, kernel_size=5, padding='same')
bn21 = tf.layers.batch_normalization(inputs=y21_conv, axis=-1, momentum=0.99, epsilon=0.001)
y22_conv = tf.layers.conv2d(inputs=bn21, filters=25, kernel_size=5, padding='same')
bn22 = tf.layers.batch_normalization(inputs=y22_conv, axis=-1, momentum=0.99, epsilon=0.001)
y22_flat = tf.contrib.layers.flatten(inputs=bn22)
y23 = tf.layers.dense(inputs=y22_flat, units=250, activation=tf.nn.relu)
y24 = tf.layers.dense(inputs=y23, units=num_jung)
y_hat_jung = tf.nn.softmax(y24)

#for jong
y31_conv = tf.layers.conv2d(inputs=x, filters=10, kernel_size=5, padding='same')
bn31 = tf.layers.batch_normalization(inputs=y31_conv, axis=-1, momentum=0.99, epsilon=0.001)
y32_conv = tf.layers.conv2d(inputs=bn31, filters=25, kernel_size=5, padding='same')
bn32 = tf.layers.batch_normalization(inputs=y32_conv, axis=-1, momentum=0.99, epsilon=0.001)
y32_flat = tf.contrib.layers.flatten(inputs=bn32)
y33 = tf.layers.dense(inputs=y32_flat, units=250, activation=tf.nn.relu)
y34 = tf.layers.dense(inputs=y33, units=num_jong)
y_hat_jong = tf.nn.softmax(y34)

#define label
y_label_cho = tf.slice(y_label, [0, 0], [-1, num_cho])
y_label_jung = tf.slice(y_label, [0, num_cho], [-1, num_jung])
y_label_jong = tf.slice(y_label, [0, num_cho + num_jung], [-1, -1])

# Define loss and optimizer
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_hat))
loss1 = tf.reduce_mean(-tf.reduce_sum(y_label_cho * tf.log(y_hat_cho), axis=[1]))
loss2 = tf.reduce_mean(-tf.reduce_sum(y_label_jung * tf.log(y_hat_jung), axis=[1]))
loss3 = tf.reduce_mean(-tf.reduce_sum(y_label_jong * tf.log(y_hat_jong), axis=[1]))
train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2)
train_step3 = tf.train.AdamOptimizer(0.0001).minimize(loss3)

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
"""
# tensorboard summary
cost_summary = tf.summary.scalar('cost', loss)
accuracy_summary_1 = tf.summary.scalar('accuracy_cho', accuracy[0])
accuracy_summary_2 = tf.summary.scalar('accuracy_jung', accuracy[1])
accuracy_summary_3 = tf.summary.scalar('accuracy_jong', accuracy[2])
merged = tf.summary.merge_all()
"""
# Start session
sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('tensorboard/train', sess.graph)
tf.global_variables_initializer().run()

# Train
print('=====Training=====')
for it in range(max_epoch):
    print("Iteration %d/%d" % (it + 1, max_epoch))
    batch_xs, batch_ys = data.train.next_batch(batch_size)
    result = sess.run([train_step1, train_step2, train_step3] + accuracy, feed_dict={x: batch_xs, y_label: batch_ys})
    #print('Mean Cost : %f' % result[1])
    print('Train Accuracy : %f, %f, %f' % (result[3], result[4], result[5]))
"""    if it % 10 == 0:
        train_writer.add_summary(merged.eval(feed_dict={x: batch_xs, y_label: batch_ys}), it)
"""
# Test
# print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))
#test_data = load_data('unicode_SeoulNamsanB')
feed_dict_test = {x: data.test.images, y_label: data.test.labels}
print('\nTest Accuracy : %f, %f, %f' % tuple(sess.run(accuracy, feed_dict=feed_dict_test)))
