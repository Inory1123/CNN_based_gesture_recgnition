import numpy as np
import tensorflow as tf
import res.utils as utils
import matplotlib.pyplot as plt

x_data = np.linspace(-1,1,300,dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name ='y_input')

l1 = utils.add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = utils.add_layer(l1,10,1,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
lines = ax.plot(x_data, y_data, 'r-', lw=5)
plt.ion()
plt.show()

with tf.Session() as sess:
    sess.run(init)
    #writer = tf.summary.FileWriter('/home/inory/graphs', sess.graph)
    for step in range(1000):
        sess.run(train,feed_dict={xs:x_data,ys:y_data})
        if step % 100 == 0:
            ax.lines.remove(lines[0])
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(1)