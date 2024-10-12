import tensorflow.compat.v1 as tf

x = tf.placeholder(tf.float32, shape=())


def true_fn():
    return tf.multiply(x, 10)


def false_fn():
    return tf.add(x, 10)


y = x % 2
z = tf.cond(tf.equal(y, 0), true_fn, false_fn)

with tf.Session() as sess:
    print(sess.run(z, feed_dict={x: 2}))  # 输出 10 (2 * 10)
    print(sess.run(z, feed_dict={x: 1}))  # 输出 11 (1 + 10)
