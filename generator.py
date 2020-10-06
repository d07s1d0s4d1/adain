import os
import itertools
import random
import numpy as np
import math
from tensorflow.keras.preprocessing import image


def gen(name, size=(256, 256), batch_size=8, mode='cartesian', in_memory=False):
    c_name = name + '/content/'
    s_name = name + '/style/'
    c_files = os.listdir(c_name)
    s_files = os.listdir(s_name)

    if mode == 'simple':
        #assert len(c_files)==len(s_files)
        l = min(len(c_files), len(s_files))
        iter_func = zip
    elif mode == 'cartesian':
        l = len(c_files) * len(s_files)
        iter_func = itertools.product

    if in_memory:
        c_files = [image.img_to_array(image.load_img(c_name + cf, target_size=size)) for cf in c_files]
        s_files = [image.img_to_array(image.load_img(s_name + sf, target_size=size)) for sf in s_files]
        get = lambda folder, x: x
    else:
        get = lambda folder, f: image.img_to_array(image.load_img(folder + f, target_size=size))
    
    while True:
        random.shuffle(c_files)
        random.shuffle(s_files)
        it = iter_func(c_files, s_files)
        
        for i in range(math.ceil(l/batch_size)):
            c_batch = []
            s_batch = []
            for _ in range(min( batch_size, l - i*batch_size )):
                c, s = next(it)
                c_batch.append(get(c_name, c))
                s_batch.append(get(s_name, s))

            X1 = np.array(c_batch) / 255.
            X2 = np.array(s_batch) / 255.

            yield [X1, X2], X1
