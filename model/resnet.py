import tensorflow as tf

try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

# Only for 18 or 34 layers
class BasicBlock(keras.Model):
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=3,
                                         padding='same',
                                         strides=stride,
                                         kernel_initializer='he_normal')
        self.bn1 = keras.layers.BatchNormalization(axis=-1)
        self.relu = keras.layers.ReLU()
        self.conv2 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=3,  
                                         padding='same',
                                         kernel_initializer='he_normal')
        self.bn2 = keras.layers.BatchNormalization(axis=-1)
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, **kwargs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        x = keras.layers.add([x, residual])
        x = self.relu(x)

        return x


# for 50, 101 or 152 layers
class Bottleneck(keras.Model):
    expansion = 4  

    def __init__(self, filters, stride=1, downsample=None):
        super(Bottleneck,self).__init__()

        self.conv1 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=1,
                                         strides=stride,
                                         kernel_initializer='he_normal'
                                         )
        self.bn1 = keras.layers.BatchNormalization(axis=3)
        self.conv2 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=3,
                                         padding='same',
                                         kernel_initializer='he_normal')
        self.bn2 = keras.layers.BatchNormalization(axis=3)
        self.conv3 = keras.layers.Conv2D(filters=filters*4,
                                         kernel_size=1,
                                         kernel_initializer='he_normal')
        self.bn3 = keras.layers.BatchNormalization(axis=3)
        self.relu = keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, training=None, mask=None):
        residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        x = keras.layers.add([x, residual])
        x = self.relu(x)

        return x


class ResNet(keras.Model):
    def __init__(self, block, layers, numclasses=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.padding = keras.layers.ZeroPadding2D((3, 3))
        self.conv1 = keras.layers.Conv2D(filters=64,
                                         kernel_size=7,
                                         strides=2,
                                         kernel_initializer='glorot_uniform')
        #  [batch, height, width, channel]
        self.bn1 = keras.layers.BatchNormalization(axis=3)
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D((3, 3), strides=2, padding='same')
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(numclasses, activation='softmax')

    def _make_layer(self, block, filter, blocks, stride=1):
        """blocks: e.g. [3, 4, 6, 3]"""
        downsample = None
        if stride != 1 or self.inplanes != filter * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2D(filters=filter * block.expansion, kernel_size=1, strides=stride),
                keras.layers.BatchNormalization(axis=3)]
            )

        layer = keras.Sequential()
        layer.add(block(filters=filter, stride=stride, downsample=downsample))

        for i in range(1, blocks):
            layer.add(block(filters=filter))

        return layer

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # conv_2_x
        x = self.layer1(x)
        # conv_3_x
        x = self.layer2(x)
        # conv_4_x
        x = self.layer3(x)
        # conv_5_x
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    model = resnet101()
    model.build(input_shape=(None, 224, 224, 3))
    model.padding.build(input_shape=(None, 224, 224, 3))
    model.summary()
