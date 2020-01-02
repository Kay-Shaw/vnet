"""
Version: 1.0.4      增加了能同时存在预训练参数与随机初始化参数的功能
Version: 1.0.3      修复了残差连接不可用的bug，并引入了深度可分离卷积和参数冻结
Version: 1.0.2      引入了膨胀卷积，残差连接，重写卷积类
Version: 1.0.1      引入了反卷积，三维卷积
Version: 1.0.0      对 tensorflow 进行简单封装
"""
import tensorflow as tf


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
                 res,
                 atrous,
                 separable,
                 transpose,
                 dropout,
                 dropout_rate,
                 parameter):
        """
        初始化卷积层参数

        :param input_dim:       输入维度
        :param output_dim:      输出维度，当使用深度可分离卷积时，这个参数表示输出是输入通道的倍数
        :param kernel_size:     卷积核尺寸
        :param conv_stride:     卷积步长，当使用膨胀卷积时，这个参数表示膨胀率
        :param conv_padding:    卷积填充
        :param input_tensor:    输入张量
        :param activate_type:   激活类型，包括三种："sigmoid", "relu" 和 "linear"
        :param para_name:       参数张量名称
        :param pool_size:       池化核尺寸，默认最大值池化
        :param pool_stride:     池化步长
        :param pool_padding:    池化填充
        :param batch_normalize: 是否批归一化, 选择True或False
        :param name:            卷积层张量名称
        :param res:             残差连接
        :param atrous:          是否为膨胀卷积
        :param separable:       是否为可分离卷积
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
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.res = lambda res_input: res
        self.output = None

        if not self.res(True):
            self.res = lambda res_input: tf.zeros(res_input.get_shape())

        self.activate_dict = dict()
        self.init_dict = dict()
        self.conv_dict = dict()
        self.pool_dict = dict()

        self.activate_dict["sigmoid"] = tf.nn.sigmoid
        self.activate_dict["relu"] = tf.nn.relu
        self.activate_dict["linear"] = lambda conv_value: conv_value
        self.activate_dict["None"] = lambda conv_value: conv_value

        if (self.parameter is not None) and (self.para_name in self.parameter):
            self.init_type = "constant"
            self.init_dict["constant"] = tf.constant_initializer(self.parameter[self.para_name])
        else:
            self.init_type = str(self.activate_type)

        self.init_dict["sigmoid"] = tf.contrib.layers.xavier_initializer(dtype=self.data_accuracy)
        self.init_dict["None"] = tf.contrib.layers.xavier_initializer(dtype=self.data_accuracy)
        self.init_dict["relu"] = tf.keras.initializers.he_normal()

        assert not ((atrous and self.transpose) and (separable and self.transpose) and (atrous and separable)), \
            "可分离, 膨胀和转置只能三选一"
        if separable:
            self.conv_dimension = str(conv_stride[2])
            self.conv_dict["1"] = tf.nn.depthwise_conv2d
        elif atrous:
            self.conv_dimension = str(conv_stride)
            self.conv_dict["2"] = tf.nn.atrous_conv2d
        else:
            self.conv_dimension = str(len(conv_stride)) + str(self.transpose)
            self.conv_dict["5False"] = tf.nn.conv3d
            self.conv_dict["5True"] = tf.nn.conv3d_transpose
            self.conv_dict["4False"] = tf.nn.conv2d
            self.conv_dict["4True"] = tf.nn.conv2d_transpose

        self.pool_dict["4"] = tf.nn.max_pool
        self.pool_dict["5"] = tf.nn.max_pool3d

        self.output_trans_shape = list()
        if self.transpose:
            for i in range(len(conv_stride) - 1):
                self.output_trans_shape.append(self.input_tensor.get_shape().as_list()[i] * conv_stride[i])
        self.output_trans_shape += [self.output_dim]

        self.l2_reg = tf.contrib.layers.l2_regularizer(0.001)

    def forward(self, conv_stride, pool_stride, pool_size, weight):
        with tf.name_scope(self.name):
            if self.transpose:
                Z = self.conv_dict[self.conv_dimension](self.input_tensor,
                                                        weight,
                                                        self.output_trans_shape,
                                                        conv_stride,
                                                        self.conv_padding)
            else:
                Z = self.conv_dict[self.conv_dimension](self.input_tensor,
                                                        weight,
                                                        conv_stride,
                                                        self.conv_padding)

            if self.batch_normalize:
                axis = list(range(len(Z.get_shape()) - 1))
                mean, variance = tf.nn.moments(Z, axis)
                Z = tf.nn.batch_normalization(Z, mean, variance, 0, 1, 0.001)

            Z = Z + self.res(Z)

            A = self.activate_dict[str(self.activate_type)](Z)

            if self.dropout:
                A = tf.nn.dropout(A, self.dropout_rate)

            if self.pool_size:
                self.output = self.pool_dict[str(len(conv_stride))](A,
                                                                    ksize=pool_size,
                                                                    strides=pool_stride,
                                                                    padding=self.pool_padding)
            else:
                self.output = A

        return self.output


class Tf2DConvPackage(TfCnnPackage):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_tensor,
                 para_name,
                 name,
                 res=None,
                 activate_type='relu',
                 kernel_size=3,
                 pool_size=None,
                 conv_stride=1,
                 conv_padding='SAME',
                 pool_stride=1,
                 pool_padding='SAME',
                 batch_normalize=False,
                 data_accuracy=tf.float32,
                 atrous=False,
                 transpose=False,
                 separable=False,
                 dropout=False,
                 dropout_rate=0,
                 trainable=True,
                 parameter=None):

        if atrous:
            self.nn_conv_stride = conv_stride
        else:
            self.nn_conv_stride = [1, conv_stride, conv_stride, 1]
        self.nn_pool_stride = [1, pool_stride, pool_stride, 1]
        self.nn_pool_size = [1, pool_size, pool_size, 1]

        super(Tf2DConvPackage, self).__init__(input_dim,
                                              output_dim,
                                              kernel_size,
                                              self.nn_conv_stride,
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
                                              res,
                                              atrous,
                                              separable,
                                              transpose,
                                              dropout,
                                              dropout_rate,
                                              parameter)

        # 创建参数张量变量
        if self.transpose:
            self.weight_shape = [self.kernel_size, self.kernel_size, self.output_dim, self.input_dim]
        else:
            self.weight_shape = [self.kernel_size, self.kernel_size, self.input_dim, self.output_dim]
        self.weight_variable = tf.get_variable(self.para_name,
                                               self.weight_shape,
                                               trainable=trainable,
                                               initializer=self.init_dict[self.init_type],
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
                 res=None,
                 atrous=False,
                 separable=False,
                 transpose=False,
                 dropout=False,
                 dropout_rate=0,
                 trainable=True,
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
                                              res,
                                              atrous,
                                              separable,
                                              transpose,
                                              dropout,
                                              dropout_rate,
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
                                               trainable=trainable,
                                               initializer=self.init_dict[self.init_type],
                                               regularizer=self.l2_reg,
                                               dtype=self.data_accuracy)

    def forward(self):
        predict = super(Tf3DConvPackage, self).forward(self.nn_conv_stride,
                                                       self.nn_pool_stride,
                                                       self.nn_pool_size,
                                                       self.weight_variable)
        return predict


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
            self._initializer = tf.contrib.layers.xavier_initializer(
                dtype=self.data_accuracy)
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
                self.output = tf.nn.max_pool(A,
                                             ksize=[1, self.pool_size, 1],
                                             strides=[1, self.pool_stride, 1],
                                             padding=self.pool_padding)
            else:
                self.output = A

        return self.output
