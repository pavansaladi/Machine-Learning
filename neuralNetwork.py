import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.image as mpimg
import random
import math
from numpy import exp

folderName = "Datasets/NeuralNetwork/"
learning_rate = 0.01

def sigmoid(x):
    x.reshape(3, 1)
    return 1 / (1 + exp(-x))

def sigmoid1(x):
    x.reshape(2, 1)
    return 1 / (1 + exp(-x))

  
inp = []
tn = []
#Reads input and appends it to a list
def make_input():
    cnt=0
    for file in os.listdir(folderName+"train_new"):
        print(cnt)
        image = Image.open(folderName+"train_new/"+file)
        img_resized = image.resize((50, 50))
        gray = img_resized.convert('L')
        arr = np.asarray(gray)
        inp.append(np.reshape(arr, (2500)))
        if file[0]=='c':
            tn.append(0)
        elif file[0] == 'd':
            tn.append(1)
        else:
            print("File Name Not Possible")
        cnt+=1
      
#Calculates the activation of the first hidden layer and second hidden layer
def calc_activation(layer_no, inp):
    if layer_no==1:
        val = sigmoid(np.matmul(weight1.T, inp.reshape(2500, 1)))
    if layer_no==2:
        val = sigmoid1(np.matmul(weight2.T, inp.reshape(3, 1)))
    return val
    
#Back propagates the weights
def backpropagation(activation1, activation2, tn, cnt):
    for j in range(0, 3):
        for k in range(0, 2):
            weight2[j][k] = weight2[j][k] + learning_rate*activation1[j]*(activation2[k]-tn[cnt])
    
    for i in range(0, 2500):
        for j in range(0, 3):
            val = 0
            for k in range(0, 2):
                val = val + (activation2[k] - tn[cnt])*weight2[j][k]
            weight1[i][j] = weight1[i][j] + learning_rate*inp[cnt][i]*val*activation1[j]*(1-activation1[j])
        
#Trains with the training data
def train():
    cnt=0
    for img in inp:
        print(cnt)
        activation1 = calc_activation(1, img)
        activation2 = calc_activation(2, activation1)
        backpropagation(activation1, activation2, tn, cnt)    
        cnt = cnt+1


test_inp = []
tn_test = []

#Tests with the testing data
def test():
    cnt=0
    correctly_classified = 0
    for file in os.listdir(folderName+"test_new"):
        print(cnt)
        image = Image.open(folderName+"test_new/"+file)
        img_resized = image.resize((50, 50))
        gray = img_resized.convert('L')
        arr = np.asarray(gray)
        test_inp.append(np.reshape(arr, (2500)))
        if file[0]=='c':
            tn_test.append(0)
        elif file[0] == 'd':
            tn_test.append(1)
        else:
            print("File Name Not Possible")
        cnt+=1
        
    i=0
    for img in test_inp:
        activation1_test = calc_activation(1, img)
        activation2_test = calc_activation(2, activation1_test)
        if(activation2_test[0] >= activation2_test[1]):
            print("Classified as CAT", end="\t")
            if(tn_test[i] == 0):
                print("Classified CORRECTLY")
                correctly_classified= correctly_classified + 1
            else:
                print("Classified WRONGLY")
        else:
            print("Classified as DOG", end="\t")
            if(tn_test[i] == 1):
                print("Classified CORRECTLY")
                correctly_classified+=1
            else:
                print("Classified WRONGLY")
    print("No Of Correctly Classified Images: " + correctly_classified)
    print("Accuracy: " + correctly_classified/5000)
        
     
        

if __name__ == "__main__":
    cnt=0
    correctly_classified=0
    weight1 = np.random.uniform(-1, 1, (2500, 3))
    weight2 = np.random.uniform(-1, 1, (3, 2))
    print("Making Input...")
    make_input()
    print("Training...")
    train()
    print(weight1)
    print("#############################################")
    print(weight2)
    print("Testing...")
    test()
    print("Ended...")
        
























































    
    