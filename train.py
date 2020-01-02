from tensorflow.python.framework import ops
# from readNii import *
from readingdata import *
from loss import *
# from Net import *
from simple_v_net_ori import *
# from try_read import load_data
import time
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# image_path = r"./data_nii/DICM/"
# label_path = r"./data_nii/Label/"


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("num_epochs", 200, "number of epoch")
flags.DEFINE_integer("batch_size" , 1, "size of batch")
flags.DEFINE_integer("steps_per_epoch", 200, "it just training set number")
flags.DEFINE_string("model_save_path", "./model/", "The path of save model")
FLAGS = flags.FLAGS


def main(unused_argv):
    ops.reset_default_graph()

    # ********************* --- graph start --- ********************* #
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.001,
                                               global_step=global_step,
                                               decay_steps=2000,
                                               decay_rate=0.9)

    image, label = create_placeholder([128, 128, 128], FLAGS.batch_size)  # shape为16的倍数
    y = forward_propagation(image)
    ent = cross_entropy(label, y)
    loss = dice_coef_loss(y_true=label, y_pred=y)
    # dice = dice_coef_loss(y_true=label, y_pred=y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)#FLAGS.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2)

    # ********************* --- graph end --- ********************* #
    # data = load_data(image_path,label_path)
    with tf.Session(config=config) as sess:
        sess.run(init)
        train_ent_loss = []
        train_dice_loss = []
        validation_ent_loss = []
        validation_dice_loss = []
        for epoch in range(FLAGS.num_epochs):
            count = 0

            for images, labels in read_data_train([1, 128, 128, 128, 1]):
            # for images,labels, in data.train_read_data([1, 128, 128, 128, 1],30):
                strat_train_time = time.time()
                _, temp_loss, step, temp_ent, temp_output = sess.run([optimizer, loss, global_step, ent, y],
                                                                      feed_dict={image: images, label: labels})

                print("Max: " + str(temp_output.max()))
                print("Step " + str(step))
                print("dice loss" + ": " + str(temp_loss))
                print("cross entropy" + ": " + str(temp_ent))
                count += 1

                end_train_time = time.time()
                print("Train time: " + str(end_train_time - strat_train_time) + "\n\n\n")

                if count == FLAGS.steps_per_epoch:

                    val_dice_loss = []
                    val_ent_loss = []
                    val_count = 0
                    for image_val, label_val in read_data_validation([1, 128, 128, 128, 1]):
                        val_count += 1
                        val_temp_dice_loss,val_temp_ent_loss = sess.run([loss,ent], feed_dict={image: image_val, label: label_val})
                        val_dice_loss.append(val_temp_dice_loss)
                        val_ent_loss.append(val_temp_ent_loss)
                        if val_count >= 10:
                            break
                    # print("Test set loss: " + str(sum(val_loss) / 10) + "\n")

                    validation_dice_loss.append(float(sum(val_dice_loss) / 10))
                    validation_ent_loss.append(float(sum(val_ent_loss) / 10))
                    train_dice_loss.append(temp_loss)
                    train_ent_loss.append(temp_ent)
                    print(temp_loss)
                    print(float(sum(val_ent_loss) / 10))
                    print(float(sum(val_dice_loss) / 10))
                    print("Save!\n\n")
                    saver.save(sess, FLAGS.model_save_path + "model-" + str(step) + ".cpkt")
                    break
        plt.figure()
        x = list(range(FLAGS.num_epochs))
        plt.plot(x,validation_dice_loss,label='validation dice')
        plt.plot(x,train_dice_loss,label='train dice')
        plt.plot(x,validation_ent_loss,label='validation entropy')
        plt.plot(x,train_ent_loss,label='train entropy')
        plt.title('training loss with lr decay')
        plt.xlabel("epoches")
        plt.ylabel("y axis caption")
        plt.legend()
        plt.savefig('loss_200eps_9decay_his_eql.png')


if __name__ == '__main__':
    print("Train")
    tf.app.run()
