import numpy as np
import tensorflow as tf
import os
import cv2
import res.utils as utils
import res.CNN_util as cnn_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROUND = 2
TRAIN_NUM = 4000
BATCH_SIZE = 100
LEARNING_RATE = 0.0005
HYPERPARA = '_5e-4_adam'
#'../NET/round5_1e-1.ckpt'
INPUT_NET_PATH = '../NET/round2_5e-4_flip.ckpt'
OUTPUT_NET_PATH = '../NET/round4_5e-4_flip_flip.ckpt'

images_train = np.load('../new_data/images_train.npy').astype(np.float32)
labels_train = np.load('../new_data/labels_train.npy').astype(np.float32)
images_test = np.load('../new_data/images_test.npy').astype(np.float32)
labels_test = np.load('../new_data/labels_test.npy').astype(np.float32)

costs = np.zeros(120,np.float32)
train_accu = np.zeros(120,np.float32)
test_accu = np.zeros(120,np.float32)

with tf.name_scope('input') as scope:
    xs = tf.placeholder(tf.float32,[None,4096], name='input')  #input 64*64*1
    x_image = tf.reshape(xs, [-1, 64, 64, 1], name='input_image')


with tf.name_scope('Ys') as scope:
    ys = tf.placeholder(tf.float32,[None,6], name='labels')	 #output 6




with tf.name_scope('Layer1') as scope:
#conv1
    with tf.name_scope('Conv1') as scope:
        W_conv1 = cnn_util.weight_variable([5,5,1,16],name='W_conv1')
        b_conv1 = cnn_util.bias_variable([16],name='B_conv1')
        h_conv1 =cnn_util.conv2d(x_image, W_conv1) + b_conv1  # output 64*64*16
    h_pool1=cnn_util.max_pool_2x2(tf.nn.relu(h_conv1,name='Relu1'),name='MaxPool1')						#output 32*32*16

with tf.name_scope('Layer2') as scope:
#conv2
    with tf.name_scope('Conv2') as scope:
        W_conv2=cnn_util.weight_variable([5,5,16,32],name='W_conv2')
        b_conv2=cnn_util.bias_variable([32],name='B_conv2')
        h_conv2=cnn_util.conv2d(h_pool1,W_conv2)+b_conv2 #output 32*32*32
    h_pool2=cnn_util.max_pool_2x2(tf.nn.relu(h_conv2,name='Relu2'),name='MaxPool2')			#output 16*16*32

with tf.name_scope('Layer3') as scope:
#conv3
    with tf.name_scope('Conv3') as scope:
        W_conv3=cnn_util.weight_variable([5,5,32,64],name='W_conv3')
        b_conv3=cnn_util.bias_variable([64],name='B_conv3')
        h_conv3=cnn_util.conv2d(h_pool2,W_conv3)+b_conv3 #output 16*16*64
    h_pool3=cnn_util.max_pool_2x2(tf.nn.relu(h_conv3,name='Relu3'),name='MaxPool3')			#output 8*8*64
    #flat
    h_pool3_flat=tf.reshape(h_pool3,[-1,8*8*64],name='flat')

with tf.name_scope('Fc1_WithDropOut') as scope:
#fc1
    W_fc1=cnn_util.weight_variable([8*8*64,1024],name='W_fc1')
    b_fc1=cnn_util.bias_variable([1024],name='B_fc1')
    keep_prob = tf.placeholder(tf.float32)

    h_fc1=tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1,name='Relu4')
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

with tf.name_scope('Y_pre') as scope:
#output
    W_out=cnn_util.weight_variable([1024,6],name='W_out')
    b_out=cnn_util.bias_variable([6],name='B_out')

    output = tf.matmul(h_fc1_drop,W_out)+b_out

    prediction=tf.nn.softmax((output),name='Softmax')

#compute accuracy
with tf.name_scope('Accuracy') as scope:
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sum_accu_test = tf.summary.scalar('sum_accu_test' + HYPERPARA, accuracy)
    sum_accu_train = tf.summary.scalar('sum_accu_train' + HYPERPARA, accuracy)


with tf.name_scope('Cost') as scope:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-8,1)), reduction_indices=[1]))
    sum_cost = tf.summary.scalar('Cost' + HYPERPARA, cross_entropy)
# cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
with tf.name_scope('Train') as scope:
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


merge_train = tf.summary.merge((sum_accu_train,sum_cost))

saver = tf.train.Saver()

with tf.Session() as sess:
    if INPUT_NET_PATH is not None and OUTPUT_NET_PATH is None:
        # 提取变量
        saver.restore(sess, INPUT_NET_PATH)
        cap = cv2.VideoCapture(1)

        if cap.isOpened() is False:
            print('调用camera失败')
            exit()

        while (1):
            ret, frame = cap.read()
            cv2.imshow("capture", frame)

            key = cv2.waitKey(3) & 0xFF

            if key == ord('b'):
                img_bg = cv2.GaussianBlur(frame, (5, 5), 0)

            if key == ord('p'):
                img_BGR = cv2.GaussianBlur(frame, (5, 5), 0)
                gray_img = utils.bakSubYCrCb(img_bg,img_BGR)
                cv2.imshow('gray', gray_img)
                pred = sess.run(prediction,feed_dict={xs: np.reshape(gray_img,(1,4096)), keep_prob:1})
                print(np.argmax(pred))
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    if INPUT_NET_PATH is not None and OUTPUT_NET_PATH is not None:
        # 提取变量
        saver.restore(sess, INPUT_NET_PATH)
        for i in range(ROUND):
            for j in range(40):
                batch_x = images_train[j * 100:(j + 1) * 100]
                batch_y = labels_train[j * 100:(j + 1) * 100]
                cost = sess.run(cross_entropy, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 1})
                accu = sess.run(accuracy, feed_dict={xs: images_test, ys: labels_test, keep_prob: 1})
                print('time:', i * 40 + j, 'cost', cost, '  accuracy', accu)
                sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
        saver.save(sess, OUTPUT_NET_PATH)

    if INPUT_NET_PATH is None:
        #初始化变量
        sess.run(tf.global_variables_initializer())
        # 绘制结构图
        # writer = tf.summary.FileWriter('/home/inory/PycharmProjects/graphs',sess.graph)
        for i in range(ROUND):
            for j in range(40):
                batch_x = images_train[j*100:(j+1)*100]
                batch_y = labels_train[j*100:(j+1)*100]
                cost = sess.run(cross_entropy, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 1})
                accu = sess.run(accuracy, feed_dict={xs: images_test, ys: labels_test, keep_prob: 1})
                print('time:', i * 40 + j, 'cost', cost, '  accuracy', accu)

                sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})

        # writer.close()
        # np.save("../data/cost"+HYPERPARA+".npy", costs)
        # np.save("../data/train_accu" + HYPERPARA + ".npy", train_accu)
        # np.save("../data/test_accu" + HYPERPARA + ".npy", test_accu)
        if OUTPUT_NET_PATH is not None:
            saver.save(sess,OUTPUT_NET_PATH)





