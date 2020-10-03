import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
import os
import shutil


def create_dir(name):
    os.mkdir('sample_data/' + TYPE +  '_data/' + name + '/')
    os.mkdir('sample_data/' + TYPE +  '_data/' + name + '/content/')
    os.mkdir('sample_data/' + TYPE +  '_data/' + name + '/style/')
    os.mkdir('sample_data/' + TYPE +  '_data/' + name + '/val_content/')
    os.mkdir('sample_data/' + TYPE +  '_data/' + name + '/val_style/')

def rm_dir(name):
    if os.path.isdir('sample_data/' + TYPE +  '_data/' + name + '/'):
        shutil.rmtree('sample_data/' + TYPE +  '_data/' + name + '/')

def content_sample(name, n=None):
    if n:
        sample = random.sample(os.listdir('sample_data/' + TYPE +  '_data/content/content'), k=n)
    else:
        sample = os.listdir('sample_data/' + TYPE +  '_data/content/content')

    for filename in tqdm(sample):
        shutil.copyfile('sample_data/' + TYPE +  '_data/content/content/' + filename, \
                        'sample_data/' + TYPE +  '_data/' + name + '/content/' + filename)
        
def style_sample(name, n=None, style=None):
    if style:
        if n:
            sample = random.sample(os.listdir('sample_data/' + TYPE +  '_data/style/' + style), k=n)
        else:
            sample = os.listdir('sample_data/' + TYPE +  '_data/style/' + style)

        for filename in tqdm(sample):
            shutil.copyfile('sample_data/' + TYPE +  '_data/style/' + style + '/' + filename, \
                            'sample_data/' + TYPE +  '_data/' + name + '/style/' + filename)

    else:    
        style_names = []

        for root, dirs, files in tqdm(os.walk('sample_data/' + TYPE +  '_data/style/', topdown=False)):
            for filename in files:
                style_names.append(root + '/' + filename)
        
        if n:
            sample = random.sample(style_names, k=n)
        else:
            sample = style_names
            
        for filename in tqdm(sample):
            shutil.copyfile(filename, 'sample_data/' + TYPE +  '_data/' + name + '/style/' + filename.split('/')[-1])

def create_val_data(name, cn_samples=100, sn_samples=100):
    c_val = random.sample(os.listdir('sample_data/' + TYPE +  '_data/' + name + '/content/'), k=cn_samples)
    s_val = random.sample(os.listdir('sample_data/' + TYPE +  '_data/' + name + '/style/'), k=sn_samples)

    for filename in tqdm(c_val):
        shutil.move('sample_data/' + TYPE +  '_data/' + name + '/content/' + filename, \
                      'sample_data/' + TYPE +  '_data/' + name + '/val_content/' + filename)

    for filename in tqdm(s_val):
        shutil.move('sample_data/' + TYPE +  '_data/' + name + '/style/' + filename, \
                      'sample_data/' + TYPE +  '_data/' + name + '/val_style/' + filename)

        
def print_n(name):
    print('Content: ', len(os.listdir('sample_data/' + TYPE +  '_data/' + name + '/content/')))
    print('Style: ', len(os.listdir('sample_data/' + TYPE +  '_data/' + name + '/style/')))
    print('Content: ', len(os.listdir('sample_data/' + TYPE +  '_data/' + name + '/val_content/')))
    print('Style: ', len(os.listdir('sample_data/' + TYPE +  '_data/' + name + '/val_style/')))

def reset_and_create_sample(name, cn_samples, sn_samples, style=None):
    rm_dir(name)
    create_dir(name)
    content_sample(name, cn_samples)
    style_sample(name, sn_samples, style)
    create_val_data(name)
    print_n(name)

def plot_examples(imgs, rows, columns, figsize=(16,16)):
    fig = plt.figure(figsize=figsize)
    for i, img in enumerate(imgs, 1):
        if i > rows * columns:
            break

        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()