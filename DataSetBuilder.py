import random
import pickle

class dataPoint():
    def __init__(self):
        self.y = 0
        self.category = 0 # values 0,1,2,3,4,5
        self.partition = 'train' # values train, validation, test

def main():
    learningData = []

    # Training Data
    for index in range(100000):
        point = dataPoint()
        point.y = random.gauss(2,0.2)
        point.category = random.choice([0,1])
        point.partition = 'train'
        learningData.append(point)

    for index in range(100000):
        point = dataPoint()
        point.y = random.gauss(5,2)
        point.category = random.choice([2,3])
        point.partition = 'train'
        learningData.append(point)

    for index in range(100000):
        point = dataPoint()
        point.y = random.gauss(7,2)
        point.category = random.choice([4,5])
        point.partition = 'train'
        learningData.append(point)

    # Test Data
    for index in range(100000):
        point = dataPoint()
        point.y = random.gauss(2,0.2)
        point.category = random.choice([0,1])
        point.partition = 'test'
        learningData.append(point)

    for index in range(100000):
        point = dataPoint()
        point.y = random.gauss(5,2)
        point.category = random.choice([2,3])
        point.partition = 'test'
        learningData.append(point)

    for index in range(100000):
        point = dataPoint()
        point.y = random.gauss(7,2)
        point.category = random.choice([4,5])
        point.partition = 'test'
        learningData.append(point)

    file = open('LearningDataDirty.pkl', 'wb')
    pickle.dump(learningData, file)

main()