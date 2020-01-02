import tensorflow as tf
from layer import *


class TfCnnPackage:
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 conv_stride,
                 conv_padding,
                 input_tensor,
                 activate_type,
                 para_name,
                 pool_size,
                 pool_stride,
                 pool_padding,
                 batch_normalize,
                 data_accuracy,
                 name,
                 transpose,
                 parameter=None):


        """
        初始化卷积层参数
        :param input_dim:       输入维度
        :param output_dim:      输出维度
        :param kernel_size:     卷积核尺寸
        :param conv_stride:     卷积步长
        :param conv_padding:    卷积填充
        :param input_tensor:    输入张量
        :param activate_type:   激活类型，包括三种："sigmoid", "relu" 和 "linear"
        :param para_name:       参数张量名称
        :param pool_size:       池化核尺寸，默认最大值池化
        :param pool_stride:     池化步长
        :param pool_padding:    池化填充
        :param batch_normalize: 是否批归一化, 选择True或False
        :param name:            卷积层张量名称
        :param transpose        是否为转置卷积
        :param parameter:       输入参数
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.input_tensor = input_tensor
        self.activate_type = activate_type
        self.para_name = para_name
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.batch_normalize = batch_normalize
        self.data_accuracy = data_accuracy
        self.name = name
        self.parameter = parameter
        self.transpose = transpose
        self.output = None

        if self.parameter is not None:
            # self.data_accuracy = None
            self._initializer = tf.constant_initializer(self.parameter[self.para_name])
            if self.activate_type == "sigmoid" or None:
                self._activate = tf.nn.sigmoid

            elif self.activate_type == "relu":
                self._activate = tf.nn.relu

        else:
            # 根据激活类型，选择参数初始化方式
            if self.activate_type == "sigmoid":
                self._initializer = tf.contrib.layers.xavier_initializer(dtype=self.data_accuracy)
                self._activate = tf.nn.sigmoid

            elif self.activate_type == "relu":
                self._initializer = tf.keras.initializers.he_normal()
                self._activate = tf.nn.relu

            else:
                self._initializer = tf.contrib.layers.xavier_initializer(dtype=self.data_accuracy)
                self._activate = None

        self.l2_reg = tf.contrib.layers.l2_regularizer(0.001)

    def forward(self, conv_stride, pool_stride, pool_size, weight):
        with tf.name_scope(self.name):
            if len(conv_stride) == 5:
                if self.transpose:
                    # print(self.input_tensor)
                    # print(weight)
                    Z = tf.nn.conv3d_transpose(self.input_tensor,
                                               weight,
                                               output_shape=[self.input_tensor.get_shape().as_list()[0],
                                                             self.input_tensor.get_shape().as_list()[1]*2,
                                                             self.input_tensor.get_shape().as_list()[2]*2,
                                                             self.input_tensor.get_shape().as_list()[3]*2,
                                                             self.output_dim],
                                               strides=conv_stride,
                                               padding=self.conv_padding)
                else:
                    Z = tf.nn.conv3d(self.input_tensor,
                                     weight,
                                     strides=conv_stride,
                                     padding=self.conv_padding)
            elif len(conv_stride) == 4:
                Z = tf.nn.conv2d(self.input_tensor,
                                 weight,
                                 strides=conv_stride,
                                 padding=self.conv_padding)

            if self.batch_normalize:
                axis = list(range(len(Z.get_shape()) - 1))
                mean, variance = tf.nn.moments(Z, axis)
                Z = tf.nn.batch_normalization(Z, mean, variance, 0, 1, 0.001)

            # Z = tf.nn.dropout(Z, 0.5)

            try:
                A = self._activate(Z)
            except:
                A = Z

            if self.pool_size is not None:
                if len(pool_stride) == 5:
                    P = tf.nn.max_pool3d(A, ksize=pool_size, strides=pool_stride, padding=self.pool_padding)
                elif len(pool_stride) == 4:
                    P = tf.nn.max_pool(A, ksize=pool_size, strides=pool_stride, padding=self.pool_padding)
            else:
                P = A

            self.output = P

        return P


class Tf2DConvPackage(TfCnnPackage):

    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 conv_stride,
                 conv_padding,
                 input_tensor,
                 activate_type,
                 para_name,
                 pool_size,
                 pool_stride,
                 pool_padding,
                 batch_normalize,
                 data_accuracy,
                 name,
                 transpose=False,
                 parameter=None):
        super(Tf2DConvPackage, self).__init__(self,
                                              input_dim,
                                              output_dim,
                                              kernel_size,
                                              conv_stride,
                                              conv_padding,
                                              input_tensor,
                                              activate_type,
                                              para_name,
                                              pool_size,
                                              pool_stride,
                                              pool_padding,
                                              batch_normalize,
                                              data_accuracy,
                                              name,
                                              transpose,
                                              parameter)

        self.nn_conv_stride = [1, self.conv_stride, self.conv_stride, 1]
        self.nn_pool_stride = [1, self.pool_stride, self.pool_stride, 1]
        self.nn_pool_size = [1, self.pool_size, self.pool_size, 1]

        # 创建参数张量变量
        self.weight_variable = tf.get_variable(self.para_name,
                                               [self.kernel_size,
                                                self.kernel_size,
                                                self.input_dim,
                                                self.output_dim],
                                               initializer=self._initializer,
                                               regularizer=self.l2_reg,
                                               dtype=self.data_accuracy)

    def forward(self):
        predict = super(Tf2DConvPackage, self).forward(self.nn_conv_stride,
                                                       self.nn_pool_stride,
                                                       self.nn_pool_size,
                                                       self.weight_variable)
        return predict


class Tf3DConvPackage(TfCnnPackage):

    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 conv_stride,
                 conv_padding,
                 input_tensor,
                 activate_type,
                 para_name,
                 pool_size,
                 pool_stride,
                 pool_padding,
                 batch_normalize,
                 data_accuracy,
                 name,
                 transpose=False,
                 parameter=None):
        super(Tf3DConvPackage, self).__init__(input_dim,
                                              output_dim,
                                              kernel_size,
                                              conv_stride,
                                              conv_padding,
                                              input_tensor,
                                              activate_type,
                                              para_name,
                                              pool_size,
                                              pool_stride,
                                              pool_padding,
                                              batch_normalize,
                                              data_accuracy,
                                              name,
                                              transpose,
                                              parameter)

        self.nn_conv_stride = [1, self.conv_stride, self.conv_stride, self.conv_stride, 1]
        self.nn_pool_stride = [1, self.pool_stride, self.pool_stride, self.pool_stride, 1]
        self.nn_pool_size = [1, self.pool_size, self.pool_size, self.pool_size, 1]

        # 创建参数张量变量
        if self.transpose:
            self.weight_shape = [self.kernel_size, self.kernel_size, self.kernel_size, self.output_dim, self.input_dim]
        else:
            self.weight_shape = [self.kernel_size, self.kernel_size, self.kernel_size, self.input_dim, self.output_dim]
        self.weight_variable = tf.get_variable(self.para_name,
                                               self.weight_shape,
                                               initializer=self._initializer,
                                               regularizer=self.l2_reg,
                                               dtype=self.data_accuracy)

    def forward(self):
        predict = super(Tf3DConvPackage, self).forward(self.nn_conv_stride,
                                                       self.nn_pool_stride,
                                                       self.nn_pool_size,
                                                       self.weight_variable)
        return predict


def attentiongate(x, gs, infilters, outfilters, stride, phase, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        kernalx = (1, 1, 1, infilters, outfilters)
        Wx = weight_xavier_init(shape=kernalx, n_inputs=kernalx[0] * kernalx[1] * kernalx[2] * kernalx[3],
                                n_outputs=kernalx[-1], activefunction='relu', variable_name=scope + 'Wx')
        Bx = bias_variable([kernalx[-1]], variable_name=scope + 'Bx')
        theta_x = conv3d(x, Wx, stride) + Bx

        kernalg = (1, 1, 1, infilters, outfilters)
        Wg = weight_xavier_init(shape=kernalg, n_inputs=kernalg[0] * kernalg[1] * kernalg[2] * kernalg[3],
                                n_outputs=kernalg[-1], activefunction='relu', variable_name=scope + 'Wg')
        Bg = bias_variable([kernalg[-1]], variable_name=scope + 'Bg')
        phi_g = conv3d(gs, Wg) + Bg

        add_xg = resnet_Add(theta_x, phi_g)
        act_xg = tf.nn.relu(add_xg)

        kernalpsi = (1, 1, 1, outfilters, 1)
        Wpsi = weight_xavier_init(shape=kernalpsi, n_inputs=kernalpsi[0] * kernalpsi[1] * kernalpsi[2] * kernalpsi[3],
                                  n_outputs=kernalpsi[-1], activefunction='relu', variable_name=scope + 'Wpsi')
        Bpsi = bias_variable([kernalpsi[-1]],variable_name=scope + 'Bpsi')
        psi = conv3d(act_xg, Wpsi) + Bpsi
        sigmoid_psi = tf.nn.sigmoid(psi)

        upsample_psi = upsample3d(sigmoid_psi, scale_factor=stride, scope=None)

        # Attention: upsample_psi * x
        # upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
        #                              arguments={'repnum': outfilters})(upsample_psi)
        gat_x = tf.multiply(upsample_psi, x)
        kernal_gat_x = (1, 1, 1, outfilters, outfilters)
        Wgatx = weight_xavier_init(shape=kernal_gat_x,
                                   n_inputs=kernal_gat_x[0] * kernal_gat_x[1] * kernal_gat_x[2] * kernal_gat_x[3],
                                   n_outputs=kernal_gat_x[-1], activefunction='relu', variable_name=scope + 'Wgatx')
        Bgatx = bias_variable([kernalpsi[-1]],variable_name=scope + 'Bgatx')
        gat_x_out = conv3d(gat_x, Wgatx) + Bgatx
        gat_x_out = normalizationlayer(gat_x_out, is_train=phase, height=height, width=width, image_z=image_z,
                                       norm_type='group', scope=scope)
        outdim = outfilters

    return gat_x_out,outdim


class TfDensePackage:

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_tensor,
                 activate_type,
                 pool_size,
                 pool_stride,
                 pool_padding,
                 batch_normalize,
                 flatten,
                 data_accuracy,
                 name):
        """
        初始化全连接层参数

        :param input_dim:        输入维度
        :param output_dim:       输出维度
        :param input_tensor:     输入张量
        :param activate_type:    激活类型
        :param pool_size:        池化尺寸
        :param pool_stride:      池化步长
        :param pool_padding:     池化填充
        :param batch_normalize:  批归一化
        :param flatten           flatten
        :param name:             层名称
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_tensor = input_tensor
        self.activate_type = activate_type
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.batch_normalize = batch_normalize
        self.flatten = flatten
        self.name = name
        self.data_accuracy = data_accuracy
        self.output = None

        if self.activate_type == "sigmoid":
            self._initializer = tf.contrib.layers.xavier_initializer(dtype=self.data_accuracy)
            self._activate = tf.nn.sigmoid()

        elif self.activate_type == "relu":
            self._initializer = tf.keras.initializers.he_normal()
            self._activate = tf.nn.relu()

        else:
            self._initializer = tf.keras.initializers.random_normal(dtype=self.data_accuracy)
            self._activate = None

    def forward(self):
        with tf.name_scope(self.name):

            if self.flatten:
                self.input_tensor = tf.contrib.layers.flatten(self.input_tensor)

            A = tf.contrib.layers.fully_connected(self.input_tensor,
                                                  self.output_dim,
                                                  activation_fn=self._activate,
                                                  weights_initializer=self._initializer)
            if self.pool_size is not None:
                P = tf.nn.max_pool(A,
                                   ksize=[1, self.pool_size, 1],
                                   strides=[1, self.pool_stride, 1],
                                   padding=self.pool_padding)
            else:
                P = A

            self.output = P

        return P
