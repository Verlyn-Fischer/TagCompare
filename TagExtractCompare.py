import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import random

def loadAndExtract(modelFile):
    model = torch.load(modelFile, map_location='cpu').eval()
    print(f'Model: {modelFile}')
    print()
    for x in range(6):
        weight_x = model.fc3.weight[x]
        for y in range(6):
            weight_y = model.fc3.weight[y]
            distance = torch.dist(weight_x,weight_y,2)
            print(f'{x}, {y}, {distance}')

def summarizeTrainingData():
    file = open('LearningDataDirty.pkl', 'rb')
    learningData = pickle.load(file)
    subLearningData = random.sample(learningData,10000)
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    results = [x0,x1,x2,x3,x4,x5]
    maxNum = 4000
    for point in subLearningData:
        results[int(point.category)].append(point.y)
    plt.hist(results,100)
    plt.show()


class dataPoint():
    def __init__(self):
        self.y = 0
        self.category = 0  # values 0,1,2,3,4,5
        self.partition = 'train'  # values train, validation, test

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
    # loadAndExtract('models/DirtyRun3.pth')
    summarizeTrainingData()

main()