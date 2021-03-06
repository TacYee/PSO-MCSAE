import numpy as np
import os
import pickle
from time import gmtime, strftime
from MRFN import MRFN

def log_particle_info(id, MRFN):
    file_path = os.getcwd() + '/data/{}.txt'.format(id)
    print(file_path)
    save_append_individual(MRFN, file_path)


def save_append_individual(indi, file_path):
    with open(file_path, 'a') as myfile:
        myfile.write(indi)
        myfile.write("\n")
        myfile.flush()


if __name__ =='__main__':
    log_particle_info(1, None)


