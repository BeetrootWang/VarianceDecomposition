# Generate training and testing dataset for main_v2.py
import numpy as np

np.random.seed(0)
training_size = 1000
testing_size = 1000

for i in range(10):
    training_dataset = np.random.uniform(-1.,1.,training_size)
    testing_dataste = np.random.uniform(-1.,1.,training_size)
    training_filename = 'data/training_' + str(training_size) + '_' + str(i)
    testing_filename = 'data/testing_' + str(training_size) + '_' + str(i)
    np.save(training_filename, training_dataset)
    np.save(testing_filename, testing_dataste)