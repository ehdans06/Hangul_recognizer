import tensorflow as tf
from load_data import load_data

max_epoch = 300
batch_size = 100
num_cho, num_jung, num_jong = 19, 21, 28
tensorboard = False

data = load_data('unicode_NanumBarunGothic')

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y1_conv = tf.layers.conv2d(inputs=x, filters=20, kernel_size=5, padding='same', activation=tf.nn.relu) # use_bias=False
# y1_norm = tf.layers.batch_normalization(inputs=y1_conv)
y1_pool = tf.layers.max_pooling2d(inputs=y1_conv, pool_size=2, strides=2, padding='valid')
y2_conv = tf.layers.conv2d(inputs=y1_pool, filters=50, kernel_size=5, padding='same', activation=tf.nn.relu) # use_bias=False
# y2_norm = tf.layers.batch_normalization(inputs=y2_conv)
y2_pool = tf.layers.max_pooling2d(inputs=y2_conv, pool_size=2, strides=2, padding='valid')
y2_flat = tf.contrib.layers.flatten(inputs=y2_pool)
y3 = tf.layers.dense(inputs=y2_flat, units=500, activation=tf.nn.relu)
y4 = tf.layers.dense(inputs=y3, units=68)
y_logit = y4
y_label = tf.placeholder(dtype=tf.float32, shape=[None, 68])

y_logit_cho = tf.slice(y_logit, [0, 0], [-1, num_cho])
y_logit_jung = tf.slice(y_logit, [0, num_cho], [-1, num_jung])
y_logit_jong = tf.slice(y_logit, [0, num_cho + num_jung], [-1, -1])
# y_hat = tf.concat([y_hat_cho, y_hat_jung, y_hat_jong], 1)
y_label_cho = tf.slice(y_label, [0, 0], [-1, num_cho])
y_label_jung = tf.slice(y_label, [0, num_cho], [-1, num_jung])
y_label_jong = tf.slice(y_label, [0, num_cho + num_jung], [-1, -1])

# Define loss and optimizer
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_label_cho, logits=y_logit_cho)
    + tf.nn.softmax_cross_entropy_with_logits(labels=y_label_jung, logits=y_logit_jung)
    + tf.nn.softmax_cross_entropy_with_logits(labels=y_label_jong, logits=y_logit_jong)
)
# loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y_hat), axis=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Define accuracy
accuracy_cho = tf.equal(tf.argmax(y_logit_cho, 1), tf.argmax(y_label_cho, 1))
accuracy_jung = tf.equal(tf.argmax(y_logit_jung, 1), tf.argmax(y_label_jung, 1))
accuracy_jong = tf.equal(tf.argmax(y_logit_jong, 1), tf.argmax(y_label_jong, 1))
accuracy_total = tf.logical_and(tf.logical_and(accuracy_cho, accuracy_jung), accuracy_jong)
accuracy = []
accuracy.append(tf.reduce_mean(tf.cast(accuracy_cho, tf.float32)))
accuracy.append(tf.reduce_mean(tf.cast(accuracy_jung, tf.float32)))
accuracy.append(tf.reduce_mean(tf.cast(accuracy_jong, tf.float32)))
accuracy.append(tf.reduce_mean(tf.cast(accuracy_total, tf.float32)))

# tensorboard summary
cost_summary = tf.summary.scalar('cost', loss)
tf.summary.scalar('accuracy_cho', accuracy[0])
tf.summary.scalar('accuracy_jung', accuracy[1])
tf.summary.scalar('accuracy_jong', accuracy[2])
tf.summary.scalar('accuracy', accuracy[3])
merged = tf.summary.merge_all()

# Start session
sess = tf.InteractiveSession()
if tensorboard:
    train_writer = tf.summary.FileWriter('tensorboard/train-newmodel', sess.graph)
tf.global_variables_initializer().run()

# Train
feed_dict_valid = {x: data.validation.images, y_label: data.validation.labels}
print('=====Training=====')
for it in range(max_epoch):
    batch_xs, batch_ys = data.train.next_batch(batch_size)
    feed_dict_train = {x: batch_xs, y_label: batch_ys}
    sess.run(train_step, feed_dict=feed_dict_train)

    if it % 10 == 0:
        print("Iteration %d/%d" % (it + 1, max_epoch))
        print('Validation Cost : %f' % loss.eval(feed_dict=feed_dict_valid))
        print('Validation Accuracy : %f, %f, %f -> %f' % tuple(sess.run(accuracy, feed_dict=feed_dict_valid)))
        if tensorboard:
            train_writer.add_summary(merged.eval(feed_dict=feed_dict_valid), it)

# Test
test_data = load_data('unicode_SeoulNamsanB')
feed_dict_test = {x: test_data.test.images, y_label: test_data.test.labels}
print('\nTest Accuracy : %f, %f, %f -> %f' % tuple(sess.run(accuracy, feed_dict=feed_dict_test)))
