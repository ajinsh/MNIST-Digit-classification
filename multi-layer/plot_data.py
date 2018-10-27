### Author : Ajinkya Shinde
### This code just plots the data from csv files for plotting accuracy

### Loads required packages
import matplotlib.pyplot as plt
import numpy as np


x1, y1 = np.loadtxt("train_half_α=0.9.csv",delimiter=',',unpack=True)
x2, y2 = np.loadtxt("test_half_α=0.9.csv",delimiter=',',unpack=True)
plt.plot(x1,y1, label="Training Set")
plt.plot(x2,y2, label="Testing Set")
plt.xlabel('Epochs') 
plt.ylabel('Accuracy (%) ')
plt.title('Experiment 3 - For training set = 30000')
plt.legend()
plt.show()