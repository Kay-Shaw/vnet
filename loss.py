import tensorflow as tf


def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.contrib.layers.flatten(y_true)
    y_pred_f = tf.contrib.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss_smooth(y_true, y_pred):
    return 1 - dsc(y_true, y_pred)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def cross_entropy(y_true, y_pred):
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    # a = y_true*tf.math.log(y_pred)
    # b = 1 - y_true
    # c = b * tf.math.log(1 - y_pred)
    # loss = -tf.reduce_sum(a+c,axis=[1, 2, 3, 4])
    # loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1,2,3,4]),reduction_indices =0 )

    loss = -tf.reduce_mean(y_true * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
    return loss
def weighted_entropy(inputs_prob, labels, weight, class_nums):
    """
    inputs_prob: [batch_size, class_nums]
    labels: [batch_size]
    weight: [class_nums]
    """
    labels_one_hot = tf.one_hot(labels, depth=class_nums)  # [batch_size, class_nums]
    inputs_prob_log = tf.math.log(inputs_prob)             # [batch_size, class_nums]
    weight = tf.constant(weight)
    inputs_prob_log_weighted = tf.einsum('bc,c->bc', inputs_prob_log, weight) # [batch_size, class_nums]
    weighted_entropy = tf.multiply(inputs_prob_log_weighted, labels_one_hot)
    loss = tf.reduce_mean(weighted_entropy)
    return loss

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3, 4])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[1, 2, 3, 4])
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)


if __name__ == "__main__":
    import os
    import nibabel as nib
    import numpy as np
    from readingdata import *
    from simple_v_net_ori import *
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # image_nii = nib.load(r"./test_loss/wujiantao_niftynet_out.nii.gz").get_data()
    # image_nii = np.reshape(image_nii, [1, 321, 247, 528, 1])
    # label_nii = nib.load(r"./test_loss/wujiantao_OriginalLabel_down.nii").get_data()
    # label_nii = np.reshape(label_nii, [1, 321, 247, 528, 1])
    # image = tf.placeholder(dtype=tf.float32, shape=[1, 321, 247, 528, 1])
    # label = tf.placeholder(dtype=tf.float32, shape=[1, 321, 247, 528, 1])
    # image = tf.placeholder(dtype=tf.float32, shape=[1,128,128,128, 1])
    # label = tf.placeholder(dtype=tf.float32, shape=[1,128,128,128, 1])
    # y = forward_propagation(image)
    # loss = cross_entropy(y_true=label, y_pred=y)
    # init = tf.global_variables_initializer()
    # with tf.Session() as lsess:
    #     lsess.run(init)
    #     for images, labels, file_name, affine in read_data_test([1, 128, 128, 128, 1]):
    #         temp_loss = lsess.run(loss, feed_dict={image: images, label: labels})
    #     print(temp_loss)
    import tensorflow as tf
    import numpy as np
    t=np.random.randint(0,3,[3,4])
    sess=tf.Session()
    print("Test matrix is:\n",t)
    print("现在测试tf.reduce_sum，对tensor中的元素求和")
    print("tf.reduce_sum():",sess.run(tf.reduce_sum(t)))
    print("tf.reduce_sun(axis=0):",sess.run(tf.reduce_sum(t,axis=0)))
    print("tf.reduce_sun(axis=1):",sess.run(tf.reduce_sum(t,axis=1)))
    print("-----------------------------------------------------------------------------------------------")
    print("现在测试tf.reduce_prod，对tensor中的元素求乘积")
    print("tf.reduce_prod():",sess.run(tf.reduce_prod(t)))
    print("tf.reduce_prod(axis=0):",sess.run(tf.reduce_prod(t,axis=0)))
    print("tf.reduce_prod(axis=1):",sess.run(tf.reduce_prod(t,axis=1)))
    print("tf.reduce_prod(axis=0,keep_dims=True):",sess.run(tf.reduce_prod(t,axis=0,keep_dims=True)))
    print("tf.reduce_prod(axis=1,keep_dims=True):",sess.run(tf.reduce_prod(t,axis=1,keep_dims=True)))
    print("输出提示keep_dims将从以后的TF中移除，所以下面的测试不再测试这个参数，默认为False")
    print("-----------------------------------------------------------------------------------------------")
    print("现在测试tf.reduce_min,对tensor中的元素求最小值，reduce_max参数意义相同，忽略测试")
    print("tf.reduce_min():",sess.run(tf.reduce_min(t)))
    print("tf.reduce_min(axis=0):",sess.run(tf.reduce_min(t,axis=0)))
    print("tf.reduce_min(axis=1):",sess.run(tf.reduce_min(t,axis=1)))
    print("-----------------------------------------------------------------------------------------------")
    print("现在测试tf.reduce_mean，对tensor中的元素求均值，如果tensor元素是整数，则计算结果自动只取整数部分")
    print("tf.reduce_mean():",sess.run(tf.reduce_mean(t)))
    print("tf.reduce_mean(axis=0):",sess.run(tf.reduce_mean(t,axis=0)))
    print("tf.reduce_mean(axis=1):",sess.run(tf.reduce_mean(t,axis=1)))
    print("-----------------------------------------------------------------------------------------------")
    print("现在测试tf.reduce_all,对tensor中的元素求逻辑与")
    print("tf.reduce_all():",sess.run(tf.reduce_all(t)))
    print("tf.reduce_all(axis=0):",sess.run(tf.reduce_all(t,axis=0)))
    print("tf.reduce_all(axis=1):",sess.run(tf.reduce_all(t,axis=1)))
    print("-----------------------------------------------------------------------------------------------")
    print("现在测试tf.reduce_any,对tensor中的元素求逻辑或")
    print("tf.reduce_any():",sess.run(tf.reduce_any(t)))
    print("tf.reduce_any(axis=0):",sess.run(tf.reduce_any(t,axis=0)))
    print("tf.reduce_any(axis=1):",sess.run(tf.reduce_any(t,axis=1)))
