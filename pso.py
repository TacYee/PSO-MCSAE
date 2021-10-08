from MRFN import MRFN
import os
import get_data
import tensorflow as tf
from measure import *
import numpy as np
import copy
from utils import *

class Population():
    def __init__(self, params):
        self.gbest_score = 999999.0
        self.gbest = None
        self.indi = []
        self.params = params

    def set_gbest(self, g_mrfn):
        units = g_mrfn.units
        units = copy.deepcopy(units)
        mrfn = MRFN()
        mrfn.set_units(units)
        mrfn.score = g_mrfn.score
        self.gbest_score = mrfn.score
        self.gbest = mrfn

    def init_population(self):
        for _ in range(self.params['pop_size']):
            mrfn = MRFN()
            mrfn.init(self.params['max_channel'],self.params['max_stack'])
            self.indi.append(mrfn)

    def get_pop_size(self):
        return len(self.indi)

class PSO():
    def __init__(self, params):
        self.params = params

    def init_population(self):
        pops = Population(self.params)
        pops.init_population()
        self.pops = pops

    def evaluate_fitness(self, pops, gen_no):
        f= FitnessAssignment(pops, self.params)
        f.evalue_all(gen_no)

    def begin_to_evolve(self):
        print('Begin to ...')
        self.init_population()
        for i in range(self.params['total_generation']):
            print('Begin {}/{} generation...'.format(i, self.params['total_generation']))
            self.evaluate_fitness(self.pops, i)
            self.update(i)

    def update(self, gen_no):
        # for the first generation, just update the pbest 
        if self.pops.gbest is None:
            for i in range(self.pops.get_pop_size()):
                mrfn = self.pops.indi[i]
                mrfn.set_pbest(cae)
                if mrfn.score > self.pops.gbest_score:
                    self.pops.set_gbest(mrfn)

            for i in range(self.pops.get_pop_size()):
                mrfn = self.pops.indi[i]
                log_particle_info(i, 'The {} generation...'.format(gen_no))
                log_particle_info(i, 'g_best:' + str(self.pops.gbest))
                log_particle_info(i, 'p_best:' + str(mrfn.p_best))
                log_particle_info(i, 'before:' + str(mrfn))
                mrfn.update(self.pops.gbest)
                log_particle_info(i, 'after:' + str(mrfn))
                self.pops.indi[i] = mrfn

        else:
            for i in range(self.pops.get_pop_size()):
                mrfn = self.pops.indi[i]
                log_particle_info(i, 'The {} generation...'.format(gen_no))
                log_particle_info(i, 'g_best:' + str(self.pops.gbest))
                log_particle_info(i, 'p_best:' + str(mrfn.p_best))
                log_particle_info(i, 'before:' + str(mrfn))
                mrfn.update(self.pops.gbest)
                log_particle_info(i, 'after:' + str(mrfn))
                self.pops.indi[i] = mrfn

            for i in range(self.pops.get_pop_size()):
                mrfn = self.pops.indi[i]
                if mrfn.score > mrfn.b_score:
                    mrfn.set_pbest(mrfn)
                if mrfn.score > self.pops.gbest_score:
                    self.pops.set_gbest(mrfn)

if __name__ == '__main__':
    X_train=get_data.get_X_train()
    y_train=get_data.get_y_train()
    X_test=get_data.get_X_test()
    y_test=get_data.get_y_test()

    params = {}
    params['X_train'] = X_train
    params['y_train'] = y_train
    params['X_test'] = X_test
    params['y_test'] = y_test
    params['max_channel']=5
    params['max_stack']=5
    params['pop_size'] = 50
    params['num_class'] = 8
    params['total_generation'] = 50

    params['batch_size'] = 32
    params['epochs'] = 20

    pso = PSO(params)
    pso.begin_to_evolve()