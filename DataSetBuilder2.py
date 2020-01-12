import random
import pickle

class dataPoint():
    def __init__(self):
        self.y = 0
        self.category = 0 # values 0,1,2,3,4,5

def makeBifuractedSet(setName,pointCount,valueAndDeviations):
    learningData = []

    for index in range(pointCount):
        r = random.randint(0,len(valueAndDeviations)-1)
        point = dataPoint()
        value, deviation = valueAndDeviations[r]
        point.y = random.gauss(value,deviation)
        point.category = r
        learningData.append(point)

    file = open(setName, 'wb')
    pickle.dump(learningData, file)

makeBifuractedSet('TrainClean.pkl',2000,[(2,0.2),(2,0.2),(5,0.2),(5,0.2),(7,0.2),(7,0.2)])