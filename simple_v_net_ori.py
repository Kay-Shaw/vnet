from tensorflow_package import *


def create_placeholder(shape, batch_size=1, dtype=tf.float32):
    assert len(shape) == 3
    image = tf.placeholder(dtype=dtype, shape=[batch_size, shape[0], shape[1], shape[2], 1])
    label = tf.placeholder(dtype=dtype, shape=[batch_size, shape[0], shape[1], shape[2], 1])
    print(image)
    print(label)
    return image, label


def forward_propagation(image, para=None):

    conv1 = Tf3DConvPackage(output_dim=16,
                            kernel_size=2,
                            conv_stride=2,
                            conv_padding='SAME',
                            input_tensor=image,
                            input_dim=image.shape[-1],
                            activate_type="relu",
                            para_name='W1',
                            pool_size=None,
                            pool_stride=1,
                            pool_padding='SAME',
                            batch_normalize=True,
                            data_accuracy=tf.float32,
                            name='Conv1',
                            transpose=False,
                            parameter=para)
    conv1.forward()
    # print('image',image)
    # ------------------- $$$ 128 * 128 * 128 * 16 $$$ ------------------- #
    conv2_1 = Tf3DConvPackage(output_dim=16,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv1.output,
                              input_dim=conv1.output_dim,
                              activate_type="relu",
                              para_name='W2_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv2_1',
                              transpose=False,
                              parameter=para)
    conv2_1.forward()
    # print('conv1.output',conv1.output)
    conv2_2 = Tf3DConvPackage(output_dim=32,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=conv2_1.output,
                              input_dim=conv2_1.output_dim,
                              activate_type="relu",
                              para_name='W2_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv2_2',
                              transpose=False,
                              parameter=para)
    conv2_2.forward()
    # print('conv2_1.output',conv2_1.output)
    # ------------------- $$$ 64 * 64 * 64 * 32 $$$ ------------------- #
    conv3_1 = Tf3DConvPackage(output_dim=32,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv2_2.output,
                              input_dim=conv2_2.output_dim,
                              activate_type="relu",
                              para_name='W3_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv3_1',
                              transpose=False,
                              parameter=para)
    conv3_1.forward()
    # print('conv2_2.output',conv2_2.output)
    conv3_2 = Tf3DConvPackage(output_dim=32,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv3_1.output,
                              input_dim=conv3_1.output_dim,
                              activate_type="relu",
                              para_name='W3_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv3_2',
                              transpose=False,
                              parameter=para)
    conv3_2.forward()
    # print('conv3_1.output',conv3_1.output)
    sum3 = conv3_2.output + conv2_2.output

    conv3_3 = Tf3DConvPackage(output_dim=64,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=sum3,
                              input_dim=conv3_2.output_dim,
                              activate_type="relu",
                              para_name='W3_3',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv3_3',
                              transpose=False,
                              parameter=para)
    conv3_3.forward()
    # print('sum3',sum3)
    # ------------------- $$$ 32 * 32 * 32 * 64 $$$ ------------------- #
    conv4_1 = Tf3DConvPackage(output_dim=64,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv3_3.output,
                              input_dim=conv3_3.output_dim,
                              activate_type="relu",
                              para_name='W4_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv4_1',
                              transpose=False,
                              parameter=para)
    conv4_1.forward()
    # print('conv3_3.output',conv3_3.output)
    conv4_2 = Tf3DConvPackage(output_dim=64,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv4_1.output,
                              input_dim=conv4_1.output_dim,
                              activate_type="relu",
                              para_name='W4_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv4_2',
                              transpose=False,
                              parameter=para)
    conv4_2.forward()

    sum4 = conv4_2.output + conv3_3.output

    conv4_3 = Tf3DConvPackage(output_dim=128,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=sum4,
                              input_dim=conv4_2.output_dim,
                              activate_type="relu",
                              para_name='W4_3',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv4_3',
                              transpose=False,
                              parameter=para)
    conv4_3.forward()

    # ------------------- $$$ 16 * 16 * 16 * 128 $$$ ------------------- #
    conv5_1 = Tf3DConvPackage(output_dim=128,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv4_3.output,
                              input_dim=conv4_3.output_dim,
                              activate_type="relu",
                              para_name='W5_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv5_1',
                              transpose=False,
                              parameter=para)
    conv5_1.forward()

    conv5_2 = Tf3DConvPackage(output_dim=128,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv5_1.output,
                              input_dim=conv5_1.output_dim,
                              activate_type="relu",
                              para_name='W5_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv5_2',
                              transpose=False,
                              parameter=para)
    conv5_2.forward()

    sum5 = conv5_2.output + conv4_3.output

    conv5_3 = Tf3DConvPackage(output_dim=256,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=sum5,
                              input_dim=conv5_2.output_dim,
                              activate_type="relu",
                              para_name='W5_3',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv5_3',
                              transpose=False,
                              parameter=para)
    conv5_3.forward()

    # ------------------- $$$ 8 * 8 * 8 * 256 $$$ ------------------- #
    conv6_1 = Tf3DConvPackage(output_dim=256,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv5_3.output,
                              input_dim=conv5_3.output_dim,
                              activate_type="relu",
                              para_name='W6_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv6_1',
                              transpose=False,
                              parameter=para)
    conv6_1.forward()

    conv6_2 = Tf3DConvPackage(output_dim=256,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv6_1.output,
                              input_dim=conv6_1.output_dim,
                              activate_type="relu",
                              para_name='W6_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv6_2',
                              transpose=False,
                              parameter=para)
    conv6_2.forward()

    sum6 = conv6_2.output + conv5_3.output

    conv6_t = Tf3DConvPackage(output_dim=128,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=sum6,
                              input_dim=conv6_2.output_dim,
                              activate_type="relu",
                              para_name='W6_t',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv6_t',
                              transpose=True,
                              parameter=para)
    conv6_t.forward()

    merge1 = tf.concat([conv6_t.output, conv5_2.output], axis=-1)
    print(merge1)

    # ------------------- $$$ 16 * 16 * 16 * 128 $$$ ------------------- #
    conv7_1 = Tf3DConvPackage(output_dim=128,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=merge1,
                              input_dim=conv5_2.output_dim + conv6_t.output_dim,
                              activate_type="relu",
                              para_name='W7_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv7_1',
                              transpose=False,
                              parameter=para)
    conv7_1.forward()

    conv7_2 = Tf3DConvPackage(output_dim=128,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv7_1.output,
                              input_dim=conv7_1.output_dim,
                              activate_type="relu",
                              para_name='W7_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv7_2',
                              transpose=False,
                              parameter=para)
    conv7_2.forward()

    sum7 = conv7_2.output + conv6_t.output

    conv7_t = Tf3DConvPackage(output_dim=64,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=sum7,
                              input_dim=conv7_2.output_dim,
                              activate_type="relu",
                              para_name='W7_t',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv7_t',
                              transpose=True,
                              parameter=para)
    conv7_t.forward()

    merge2 = tf.concat([conv7_t.output, conv4_2.output], axis=-1)
    print(merge2)

    # ------------------- $$$ 32 * 32 * 32 * 64 $$$ ------------------- #
    conv8_1 = Tf3DConvPackage(output_dim=64,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=merge2,
                              input_dim=conv4_2.output_dim + conv7_t.output_dim,
                              activate_type="relu",
                              para_name='W8_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv8_1',
                              transpose=False,
                              parameter=para)
    conv8_1.forward()

    conv8_2 = Tf3DConvPackage(output_dim=64,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv8_1.output,
                              input_dim=conv8_1.output_dim,
                              activate_type="relu",
                              para_name='W8_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv8_2',
                              transpose=False,
                              parameter=para)
    conv8_2.forward()

    sum8 = conv8_2.output + conv7_t.output

    conv8_t = Tf3DConvPackage(output_dim=32,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=sum8,
                              input_dim=conv8_2.output_dim,
                              activate_type="relu",
                              para_name='W8_t',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv8_t',
                              transpose=True,
                              parameter=para)
    conv8_t.forward()

    merge3 = tf.concat([conv8_t.output, conv3_2.output], axis=-1)
    print(merge3)

    # ------------------- $$$ 64 * 64 * 64 * 32 $$$ ------------------- #
    conv9_1 = Tf3DConvPackage(output_dim=32,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=merge3,
                              input_dim=conv3_2.output_dim + conv8_t.output_dim,
                              activate_type="relu",
                              para_name='W9_1',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv9_1',
                              transpose=False,
                              parameter=para)
    conv9_1.forward()

    conv9_2 = Tf3DConvPackage(output_dim=32,
                              kernel_size=5,
                              conv_stride=1,
                              conv_padding='SAME',
                              input_tensor=conv9_1.output,
                              input_dim=conv9_1.output_dim,
                              activate_type="relu",
                              para_name='W9_2',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=False,
                              data_accuracy=tf.float32,
                              name='Conv9_2',
                              transpose=False,
                              parameter=para)
    conv9_2.forward()

    sum9 = conv9_2.output + conv8_t.output

    conv9_t = Tf3DConvPackage(output_dim=16,
                              kernel_size=2,
                              conv_stride=2,
                              conv_padding='SAME',
                              input_tensor=sum9,
                              input_dim=conv9_2.output_dim,
                              activate_type="relu",
                              para_name='W9_t',
                              pool_size=None,
                              pool_stride=1,
                              pool_padding='SAME',
                              batch_normalize=True,
                              data_accuracy=tf.float32,
                              name='Conv9_t',
                              transpose=True,
                              parameter=para)
    conv9_t.forward()

    merge4 = tf.concat([conv9_t.output, conv2_1.output], axis=-1)
    print(merge4)

    # ------------------- $$$ 128 * 128 * 128 * 16 $$$ ------------------- #
    conv10_1 = Tf3DConvPackage(output_dim=16,
                               kernel_size=5,
                               conv_stride=1,
                               conv_padding='SAME',
                               input_tensor=merge4,
                               input_dim=conv2_1.output_dim + conv9_t.output_dim,
                               activate_type="relu",
                               para_name='W10_1',
                               pool_size=None,
                               pool_stride=1,
                               pool_padding='SAME',
                               batch_normalize=False,
                               data_accuracy=tf.float32,
                               name='Conv10_1',
                               transpose=False,
                               parameter=para)
    conv10_1.forward()

    sum10 = conv10_1.output + conv9_t.output

    conv10_t = Tf3DConvPackage(output_dim=10,
                               kernel_size=2,
                               conv_stride=2,
                               conv_padding='SAME',
                               input_tensor=sum10,
                               input_dim=conv10_1.output_dim,
                               activate_type="relu",
                               para_name='W10_t',
                               pool_size=None,
                               pool_stride=1,
                               pool_padding='SAME',
                               batch_normalize=False,
                               data_accuracy=tf.float32,
                               name='Conv10_t',
                               transpose=True,
                               parameter=para)
    conv10_t.forward()

    merge5 = tf.concat([conv10_t.output, image], axis=-1)

    conv11 = Tf3DConvPackage(output_dim=1,
                             kernel_size=5,
                             conv_stride=1,
                             conv_padding='SAME',
                             input_tensor=merge5,
                             input_dim=conv10_t.output_dim + 1,
                             activate_type="sigmoid",
                             para_name='W11',
                             pool_size=None,
                             pool_stride=1,
                             pool_padding='SAME',
                             batch_normalize=False,
                             data_accuracy=tf.float32,
                             name='Conv11',
                             transpose=False,
                             parameter=para)
    conv11.forward()

    return conv11.output


if __name__ == "__main__":
    image_test, label_test = create_placeholder([288, 288, 288])
    print(image_test)
    print(label_test)
    # pool_test = tf.nn.max_pool3d(image_test, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    # print(pool_test)

    result = forward_propagation(image_test)
    print(result)
