from tensorflow.python.framework import ops
# from readNii import *
from readingdata import *
from skimage import transform
from loss import *
# from simple_v_net import *
from simple_v_net_ori import *
import nibabel as nib
import copy
import os
import cv2
import matplotlib.pyplot as plt


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = r"./model/"
# model_path = r'C:/Users/37474/Desktop/vnet/model/'
save_path = r"./data_test/"


def load_parameters(pattern, model_name=None):
    parameters = {}

    if pattern == 'restore_train':
        saver = tf.train.import_meta_graph(model_path + model_name + '.meta')
    if pattern == 'demo':
        saver = tf.train.import_meta_graph(model_path + model_name + '.meta')

    init = tf.global_variables_initializer()

    with tf.Session() as sess_to_load:
        sess_to_load.run(init)

        graph = tf.get_default_graph()

        if pattern == 'restore_train':
            saver.restore(sess_to_load, model_path+model_name)
        elif pattern == 'demo':
            saver.restore(sess_to_load, model_path+model_name)

        print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])

        # parameters["W0"] = graph.get_tensor_by_name("W0:0").eval()
        # parameters["W1"] = graph.get_tensor_by_name("W1:0").eval()
        # parameters["W3"] = graph.get_tensor_by_name("W3:0").eval()
        # parameters["W4"] = graph.get_tensor_by_name("W4:0").eval()
        # parameters["W5"] = graph.get_tensor_by_name("W5:0").eval()

        return parameters


def infer(para):
    image, label = create_placeholder([32, 32, 32], 1)
    y = forward_propagation(image, para)
    # loss = dice_loss(label, y)
    loss = cross_entropy(label, y)
    y = tf.nn.sigmoid(y)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        data = load_data([1,32, 32, 32,1])
        count = 0
        result, temp_loss = sess.run([y, loss], feed_dict={image: images, label: labels})
        print(temp_loss)
        result = np.squeeze(result)
        result = nib.Nifti1Image(result, np.eye(4))
        count += 1
        nib.save(result, save_path + str(count))


def infer_read_graph_old():
    ops.reset_default_graph()
    saver = tf.train.import_meta_graph(model_path + 'model-99.cpkt.meta')
    graph = tf.get_default_graph()                                         # 获取当前图，为了后续训练时恢复变量
    input_x = graph.get_operation_by_name('Placeholder').outputs[0]
    input_y = graph.get_operation_by_name('Placeholder_1').outputs[0]
    y = graph.get_operation_by_name('Conv11/Conv3D').outputs[0]            # 获取输出变量
    # loss = cross_entropy(input_y, y)
    y = tf.nn.sigmoid(y)
    loss = dice_coef_loss(input_y, y)
    with tf.Session() as sess:
        counter = 0
        load_data = DataLoader_not_resize(1, (160, 128, 256), 'Train')
        for X_test, Y_test, file_name, affine in load_data.load_image_not_resize():
            # print("affine: " + str(affine))
            print("test label data: " + str(Y_test[0, 128, 64, 211, 0]))
            print(file_name)
            print(counter)
            other_image = nib.load("E:/test_image/" + file_name)
            other_affine = other_image.affine
            saver.restore(sess, tf.train.latest_checkpoint(model_path)) 
            result, temp_loss = sess.run([y, loss], feed_dict={input_x: X_test, input_y: Y_test})
            print("Max: " + str(result.max()))

            print(str(temp_loss) + "\n\n")
            result = np.squeeze(result[0])
            result = nib.Nifti1Image(result, other_affine)
            counter += 1
            nib.save(result, save_path + file_name)

            if counter > 200:
                print("Inference timeout.")
                break

def infer_read_graph():
    ops.reset_default_graph()
    saver = tf.train.import_meta_graph(model_path + 'model-40000.cpkt.meta')
    graph = tf.get_default_graph()                                         # 获取当前图，为了后续训练时恢复变量
    input_x = graph.get_operation_by_name('Placeholder').outputs[0]
    input_y = graph.get_operation_by_name('Placeholder_1').outputs[0]
    y = graph.get_operation_by_name('Conv11/Conv3D').outputs[0]            # 获取输出变量

    y = tf.nn.sigmoid(y)
    loss1 = dice_coef_loss(input_y, y)
    loss2 = cross_entropy(input_y, y)
    with tf.Session() as sess:
        counter = 0
        for X_test, Y_test, file_name, affine in read_data_test([1,128, 128, 128,1]):
            print(file_name)
            # print(counter)
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            result, dice_loss,cross_ent= sess.run([y, loss1,loss2], feed_dict={input_x: X_test, input_y: Y_test})
            print("Max: " + str(result.max()))
            print('min: ' + str(result.min()))
            print('dice loss: '+str(dice_loss))
            print('cross_entropy: '+str(cross_ent) + "\n\n")
            result = np.squeeze(result[0])
            result = nib.Nifti1Image(result, affine)
            counter += 1
            nib.save(result, save_path + file_name)

            if counter >= 100:
                print("Inference timeout.")
                break


if __name__ == '__main__':
    # para_test = load_parameters('demo', 'model-99.cpkt')
    # para_test_copy = copy.deepcopy(para_test)
    # print(para_test_copy)
    # print("------------------------------------\n\n\n\n\n\n\n\n\n\n-----------------------------------")
    # infer(para_test_copy)

    infer_read_graph()
