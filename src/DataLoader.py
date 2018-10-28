import numpy as np
import pandas as pd
import sys, os, re
import keras
from keras.preprocessing import sequence

data_dir = "./data"
# training data inputs: x and targets: y
x_train_path = os.path.join(data_dir, 'X_train.hdf')
y_train_path = os.path.join(data_dir, 'y_train.hdf')
# validation data inputs: x and targest: y
x_valid_path = os.path.join(data_dir, 'X_test.hdf')
y_valid_path = os.path.join(data_dir, 'y_test.hdf')
# create file path for csv file with metadata about variables
metadata = os.path.join(data_dir, 'ehr_features.csv')

class DataLoader:

    def __init__(self):
        self._read()
        self._normalize()
        self._forward_fill()
        self._sequence()

    def _read(self):
        # training data
        self.X_train = pd.read_hdf(x_train_path)
        self.y_train = pd.read_hdf(y_train_path)
        # validation data
        self.X_valid = pd.read_hdf(x_valid_path)
        self.y_valid = pd.read_hdf(y_valid_path)

    def keys(self):
        # transform the column keys to avoid invalid key forms
        return self.fillvars.tolist()

    def _normalize(self):
        # read in variables from csv file (using pandas) since each varable there is tagged with a category
        self.variables = pd.read_csv(metadata, index_col=0)
        # next, select only variables of a particular category for normalization
        self.normvars = self.variables[self.variables['type'].isin(['Interventions', 'Labs', 'Vitals'])]
        # finally, iterate over each variable in both training and validation data
        for vId, dat in self.normvars.iterrows():
            self.X_train[vId] = self.X_train[vId] - dat['mean']
            self.X_valid[vId] = self.X_valid[vId] - dat['mean']
            self.X_train[vId] = self.X_train[vId] / (dat['std'] + 1e-12)
            self.X_valid[vId] = self.X_valid[vId] / (dat['std'] + 1e-12)

    def _forward_fill(self):
        # first select variables which will be filled in
        self.fillvars = self.variables[self.variables['type'].isin(['Vitals', 'Labs'])].index
        # next forward fill any missing values with more recently observed value
        self.X_train = self.X_train.groupby(level=0)[self.fillvars].ffill()[self.fillvars]
        self.X_valid = self.X_valid.groupby(level=0)[self.fillvars].ffill()[self.fillvars]
        # finally, fill in any still missing values with 0 (i.e. values that could not be filled forward)
        self.X_train.fillna(value=0, inplace=True)
        self.X_valid.fillna(value=0, inplace=True)

    def _sequence(self):
        # max number of sequence length
        maxlen = 500

        # get a list of unique patient encounter IDs
        teId = self.X_train.index.levels[0]
        veId = self.X_valid.index.levels[0]

        # pad every patient sequence with 0s to be the same length,
        # then transforms the list of sequences to one numpy array
        self.X_train = [self.X_train.loc[patient].values for patient in teId]
        self.y_train = [self.y_train.loc[patient].values for patient in teId]

        self.X_train = sequence.pad_sequences(self.X_train,
                                              dtype='float32',
                                              maxlen=maxlen,
                                              padding='post',
                                              truncating='post')
        self.y_train = sequence.pad_sequences(self.y_train,
                                              dtype='float32',
                                              maxlen=maxlen,
                                              padding='post',
                                              truncating='post')

        # repeat for the validation data
        self.X_valid = [self.X_valid.loc[patient].values for patient in veId]
        self.y_valid = [self.y_valid.loc[patient].values for patient in veId]

        self.X_valid = sequence.pad_sequences(self.X_valid,
                                              dtype='float32',
                                              maxlen=maxlen,
                                              padding='post',
                                              truncating='post')
        self.y_valid = sequence.pad_sequences(self.y_valid,
                                              dtype='float32',
                                              maxlen=maxlen,
                                              padding='post',
                                              truncating='post')
        self.X_train = np.reshape(self.X_train, newshape=(self.X_train.shape[0] * self.X_train.shape[1], self.X_train.shape[2]))
        self.X_valid = np.reshape(self.X_valid, newshape=(self.X_valid.shape[0] * self.X_valid.shape[1], self.X_valid.shape[2]))
        self.y_train = np.reshape(self.y_train, newshape=(self.y_train.shape[0] * self.y_train.shape[1], self.y_train.shape[2]))
        self.y_valid = np.reshape(self.y_valid, newshape=(self.y_valid.shape[0] * self.y_valid.shape[1], self.y_valid.shape[2]))

        if self.X_train.shape[0] == self.y_train.shape[0] and self.X_valid.shape[0] == self.y_valid.shape[0]:
            print("Successfully loaded data. We have ({}, {}) columns".format(self.X_train.shape[1], self.y_train.shape[1]))
        else:
            print("Error : The first dimension of the data doesn't match")

    def sample_batch(self, batch_size=500, mode="train"):
        # if we sample from the training data
        if mode == "train":
            index = np.random.randint(0, self.X_train.shape[0] - 1, batch_size)
            return self.X_train[index], self.y_train[index]
        elif mode == "valid":
            index = np.random.randint(0, self.X_valid.shape[0] - 1, batch_size)
            return self.X_valid[index], self.y_valid[index]
        else:
            print("Error : Invalid mode, choose between \"train\" and \"valid\"")
            return -1