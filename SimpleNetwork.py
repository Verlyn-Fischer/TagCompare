import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import random
import TagCompareUtil
import numpy as np

class dataPoint():
    def __init__(self):
        self.y = 0
        self.category = 0 # values 0,1,2,3,4,5
        self.partition = 'train' # values train, validation, test


class Trainer():
    def __init__(self,experiment,dataFile):

        # Hyper Parameters
        self.minibatch_size = 1024
        self.initial_weights_setting = 0.3
        self.learning_rate = 0.0005
        self.total_iterations = 600
        self.iterations_between_test = 2
        self.iterations_in_test = 500
        self.learning_data = []
        self.model = Net()
        self.experiment = experiment
        self.dataFile = dataFile

    def init_weights(self):
        if type(self.model) == nn.Conv2d or type(self.model) == nn.Linear:
            torch.nn.init.uniform_(self.model.weight, -1 * self.initial_weights_setting, self.initial_weights_setting)
            self.model.bias.data.fill_(self.initial_weights_setting)

    def loadTrainingData(self):

        filename = self.dataFile
        if os.path.exists(filename):
            with open(filename, 'rb') as fileObj:
                self.learning_data = pickle.load(fileObj)

    def makeOneHotCat(self,category):
        output = np.zeros(6,dtype=float)
        output[category] = 1
        output = torch.tensor(output,requires_grad=True,dtype=float)
        return output

    def buildBatch(self):
        y_batch = []
        cat_batch = []
        list_size = 0
        random.shuffle(self.learning_data)
        for point in self.learning_data:
            if point.partition == 'train':
                y_batch.append(torch.tensor(point.y,requires_grad=True,dtype=float).unsqueeze(0).unsqueeze(0).float())
                cat_item = self.makeOneHotCat(point.category).unsqueeze(0).float()
                cat_batch.append(cat_item)
                list_size += 1
            if list_size == self.minibatch_size:
                break
        y_batch = torch.cat(y_batch,0)
        cat_batch = torch.cat(cat_batch,0)
        return y_batch, cat_batch

    def train(self):
        self.model.train()
        losses = []

        # define Adam optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # initialize mean squared error loss
        criterion = nn.MSELoss()

        for iteration in range(self.total_iterations):
            if iteration % self.iterations_between_test == 0 and iteration != 0:
                ave_loss = np.array(losses).mean()
                TagCompareUtil.writeLoss(ave_loss,iteration,self.experiment)
                TagCompareUtil.writeAccuracy(self.test(),iteration,self.experiment)
                losses.clear()

            y_batch_tensor, cat_batch_tensor = self.buildBatch()

            cat_pred_batch = self.model(y_batch_tensor)
            cat_pred_batch = F.softmax(cat_pred_batch)

            # calculate loss (input, target)
            loss = criterion(cat_pred_batch, cat_batch_tensor) # per MSELoss documentation of input, target
            losses.append(loss.item())

            # do backward pass
            loss.backward()

            # do step of training
            optimizer.step()

    def test(self):
        self.model.eval()

        results = []
        for index in range(self.iterations_in_test):
            foundTest = False
            while not foundTest:
                point = random.choice(self.learning_data)
                if point.partition == 'test':
                    foundTest = True
                    y_item_tensor = torch.tensor(point.y, requires_grad=False, dtype=float).unsqueeze(0).unsqueeze(0).float()
                    cat_item = self.makeOneHotCat(point.category).unsqueeze(0).float()
                    y_pred = self.model(y_item_tensor)
                    if torch.argmax(cat_item) == torch.argmax(y_pred):
                        results.append(1)
                    else:
                        results.append(0)

        accuracy = np.array(results).mean()
        self.model.train()
        return accuracy

class Net(nn.Module):

    # This model takes an observation of the environment and predicts a reward for each possible action

    def __init__(self):
        super(Net, self).__init__()

        observation_width = 1
        connect_1_2 = 6
        connect_2_3 = 12
        connect_3_4 = 6

        self.fc1 = nn.Linear(in_features=observation_width, out_features=connect_1_2)
        self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=connect_1_2, out_features=connect_2_3)
        self.relu2 = nn.ReLU(inplace=True)
        # self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=connect_2_3, out_features=connect_3_4)


    def forward(self, x):

        out = x
        out = self.fc1(out)
        out = self.relu1(out)
        # out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        # out = self.dropout2(out)
        out = self.fc3(out)

        return out

def main():

    # Settings
    experiment = 'DirtyRun3'
    myTraininer = Trainer(experiment,'LearningDataDirty.pkl')
    startModelFile = 'models/' + 'DirtyRun3' + '.pth'

    # No need to modify
    saveModelFile = 'models/' + experiment + '.pth'
    if os.path.exists(startModelFile):
        myTraininer.model = torch.load(startModelFile, map_location='cpu').eval()
    else:
        myTraininer.init_weights()
    myTraininer.loadTrainingData()
    myTraininer.train()
    torch.save(myTraininer.model, saveModelFile)

main()
