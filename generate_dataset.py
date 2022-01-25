# Generate training and testing dataset for main_v2.py
import numpy as np

np.random.seed(0)
data_dim = 10
training_size = 10000
testing_size = 1000

for i in range(10):
    training_dataset = np.random.uniform(-1.,1.,[training_size, data_dim])
    testing_dataste = np.random.uniform(-1.,1.,[testing_size, data_dim])
    training_filename = 'data/training_' + str(training_size) + '_' + str(i)
    testing_filename = 'data/testing_' + str(testing_size) + '_' + str(i)
    np.save(training_filename, training_dataset)
    np.save(testing_filename, testing_dataste)