import numpy as np
class ConvLayer():#In order to obtain the parameters
    def __init__(self, kernel, feature_size, stride=1):
        self.kernel = kernel
        self.feature_size = feature_size
        self.stride = stride
        self.v = np.array([0.0, 0.0])
        self.type = 1

    def __str__(self):
        _str = 'C[K:{}-F:{}]'.format(self.kernel, self.feature_size)
        return _str