import numpy as np
from layers import ConvLayer
import copy
from numpy.random import seed
seed(1)
class MRFN():

    def __init__(self):
        self.score = 0.00
        self.b_score = 99999999.0

        self.w = 0.72984
        self.c1 = 1.193
        self.c2 = 1.193
        self.units = []
        self.conv_feature_size_range = [20, 100]
        self.conv_feature_size_maxv = self.conv_feature_size_range[1] - self.conv_feature_size_range[0]
        self.conv_kernel_range = [1, 5]
        self.conv_kernel_maxv = self.conv_kernel_range[1] - self.conv_kernel_range[0]
        self.conv_channel_range= [1, 5]
        self.conv_channel_maxv= self.conv_channel_range[1] - self.conv_channel_range[0]
        self.num_channel=0
        self.num_stack=0

    def reset_state(self):
        self.score = 0.00

    def set_units(self, new_units):
        self.units = new_units

    def set_pbest(self, p_mrfn):
        units = p_mrfn.units
        units = copy.deepcopy(units)
        mrfn = MRFN()
        mrfn.set_units(units)
        mrfn.score = p_mrfn.score
        self.b_score = mrfn.score
        self.p_best = mrfn
    
    def get_length(self):
        return len(self.units)

    def random_a_conv(self):#Randomly generate a convolutional layer
        kernel = self.random_conv_kernel_size()
        feature_size = self.random_conv_feature_size()
        conv = self.create_a_conv(kernel, feature_size, 1)
        return conv

    def random_conv_feature_size(self):#Randomly generate the number of feature maps
        feature_size = self.randint(self.conv_feature_size_range[0], self.conv_feature_size_range[1])
        return feature_size

    def random_conv_kernel_size(self):#Randomly generate the kernel size
        kernel = self.randint(self.conv_kernel_range[0], self.conv_kernel_range[1])
        return kernel

    def create_a_conv(self, kernel, feature_size, stride):#Generate convolutional layer
        conv = ConvLayer(kernel, feature_size, stride)
        return conv

    def randint(self, low, high):#Generate random integers for kernel size and channel
        return np.random.random_integers(low, high).item()

    def rand(self):#Generate random number for r
        return np.random.random()

    def adjust_kernel(self, kernel):
        if kernel < self.conv_kernel_range[0]:
            kernel = self.conv_kernel_range[0]
        elif kernel > self.conv_kernel_range[1]:
            kernel = self.conv_kernel_range[1]
        return int(kernel)

    def adjust_feature_size(self, feature_size):
        if feature_size < self.conv_feature_size_range[0]:
            feature_size = self.conv_feature_size_range[0]
        elif feature_size > self.conv_feature_size_range[1]:
            feature_size = self.conv_feature_size_range[1]
        return int(feature_size)

    def adjust_kernel_v(self, kernel_new_v):
        if np.abs(kernel_new_v) > self.conv_kernel_maxv:
            kernel_new_v =  (kernel_new_v/np.abs(kernel_new_v))*self.conv_kernel_maxv
        return kernel_new_v
    
    def adjust_feature_size_v(self, feature_size_new_v):
        if np.abs(feature_size_new_v) > self.conv_feature_size_maxv:
            feature_size_new_v = (feature_size_new_v/np.abs(feature_size_new_v))*self.conv_feature_size_maxv
        return feature_size_new_v

    def update(self, g_best):#x-reference update
        p_best_units = self.p_best.units#parameters of pbest
        g_best_units = g_best.units
        current_u = self.units#All three of the above need to be called before

        g_best_conv_list = []
        for i in range(len(g_best_units)):
            g_best_conv_list.append(g_best_units[i])

        p_best_conv_list = []
        for i in range(len(p_best_units)):
            p_best_conv_list.append(p_best_units[i])

        current_conv_list = []
        v_list = []#speed
        for i in range(len(current_u)):
            current_conv_list.append(current_u[i])
            v_list.append(current_u[i].v)

        new_unit_list = []
        min_length = min(len(g_best_conv_list), len(p_best_conv_list))
        for i in range(min_length):
            gbest_unit = g_best_conv_list[i]
            gbest_kernel = gbest_unit.kernel
            gbest_feature_size = gbest_unit.feature_size

            pbest_unit = p_best_conv_list[i]
            pbest_kernel = pbest_unit.kernel
            pbest_feature_size = pbest_unit.feature_size

            current_unit = current_conv_list[i]
            current_unit_kernel = current_unit.kernel
            current_unit_feature_size = current_unit.feature_size

            v_current = v_list[i]
            kernel_old_v = v_current[0]
            feature_size_old_v = v_current[1]

            kernel_new_v = self.w*kernel_old_v + self.c1*self.rand()*(gbest_kernel-current_unit_kernel) + self.c2*self.rand()*(pbest_kernel-current_unit_kernel)
            kernel_new_v = self.adjust_kernel_v(kernel_new_v)
            feature_size_new_v = self.w*feature_size_old_v + self.c1*self.rand()*(gbest_feature_size-current_unit_feature_size) + self.c2*self.rand()*(pbest_feature_size-current_unit_feature_size)
            feature_size_new_v = self.adjust_feature_size_v(feature_size_new_v)


            new_v_list = [kernel_new_v, feature_size_new_v,]
            new_kernel = self.adjust_kernel(current_unit_kernel + kernel_new_v)
            new_feature_size = self.adjust_feature_size(current_unit_feature_size + feature_size_new_v)


            new_unit = self.create_a_conv(new_kernel, new_feature_size, stride=1)
            new_unit.v = new_v_list
            new_unit_list.append(new_unit)

        if min_length < len(p_best_conv_list):
            for i in range(min_length, len(p_best_conv_list)):
                pbest_unit = p_best_conv_list[i]
                pbest_kernel = pbest_unit.kernel
                pbest_feature_size = pbest_unit.feature_size

                current_unit = current_conv_list[i]
                current_unit_kernel = current_unit.kernel
                current_unit_feature_size = current_unit.feature_size

                v_current = v_list[i]
                kernel_old_v = v_current[0]
                feature_size_old_v = v_current[1]

                kernel_new_v = self.w*kernel_old_v + self.c2*self.rand()*(pbest_kernel-current_unit_kernel)
                kernel_new_v = self.adjust_kernel_v(kernel_new_v)
                feature_size_new_v = self.w*feature_size_old_v + self.c2*self.rand()*(pbest_feature_size-current_unit_feature_size)
                feature_size_new_v = self.adjust_feature_size_v(feature_size_new_v)


                new_v_list = [kernel_new_v, feature_size_new_v]
                new_kernel = self.adjust_kernel(current_unit_kernel + kernel_new_v)
                new_feature_size = self.adjust_feature_size(current_unit_feature_size + feature_size_new_v)

                new_unit = self.create_a_conv(new_kernel, new_feature_size, stride=1)
                new_unit.v = new_v_list
                new_unit_list.append(new_unit)
        self.units = new_unit_list

    def init(self, max_channel,max_stack):#obtain the parameters of each layer
        num_channel = np.random.randint(1, max_channel)
        num_stack = np.random.randint(1,max_stack)
        self.num_channel=num_channel
        self.num_stack=num_stack
        for i in range(num_channel):
            for j in range(num_stack):
                conv = self.random_a_conv()
                self.units.append(conv)

    def __str__(self):
        _str = []
        _str.append('len:{}'.format(self.get_length()))
        _str.append('score:{:.2E}'.format(self.score))
        for u in self.units:
            _str.append(str(u))
        return ' '.join(_str)

if __name__ == '__main__':
    g_best = MRFN()
    u1 = g_best.create_a_conv(kernel=1, feature_size=54, stride=1)
    u2 = g_best.create_a_conv(kernel=2, feature_size=48, stride=1)
    u3 = g_best.create_a_conv(kernel=3, feature_size=32, stride=1)
    g_best.set_units([ u1, u2, u3])

    p_best = MRFN()
    u1 = p_best.create_a_conv(kernel=4, feature_size=64, stride=1)
    u2 = p_best.create_a_conv(kernel=5, feature_size=32, stride=1)
    p_best.set_units([u1, u2])

    current = MRFN()
    u1 = current.create_a_conv(kernel=7, feature_size=64, stride=1)
    u2 = current.create_a_conv(kernel=8, feature_size=32, stride=1)
    current.set_units([u1, u2])
    current.set_pbest(p_best)
    for i in range(5):
        current.update(g_best)
        for j in range(2):
            print(current.units[j].v)
        print(current)