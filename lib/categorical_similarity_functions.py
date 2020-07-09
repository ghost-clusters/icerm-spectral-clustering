import numpy as np
import scipy as sp
from scipy import linalg
import unittest
import csv

# Function to take csv file of categorical data and transform it into workable data
# Outputs the number of attributes for each categorical variable and the data without headings as a numpy array
def categorical_preprocessing_csv(datacsv):
    with open(datacsv, 'r') as csvfile:
        so = csv.reader(csvfile, delimiter=',', quotechar='"')
        so_data = []
        for row in so:
            so_data.append(row)
    data = so_data[1:]
    data = np.array(data)
    n,d = data.shape
    numOfAtts = np.zeros((d,))
    for j in range(d):
        values = []
        for i in range(n):
            if i == 0:
                values.append(data[0][j])
            elif (not data[i][j] in(values)):
                values.append(data[i][j])
        numOfAtts[j] = len(values)
    return numOfAtts, data

# function to craft a similarity matrix using the Eskin Similarity Function
# Takes in the number of attributes per variable and the data
# outputs similarity matrix
def eskin_similarity(numOfAtts,data):
    n,d = data.shape
    similarity_matrix = np.zeros((n,n))
    for i in range(n):
        Sx_y = np.zeros((d,))
        for j in range(n):
            for k in range(d):
                if data[i][k] == data[j][k]:
                    Sx_y[k] = 1
                else:
                    n_k = numOfAtts[k]
                    Sx_y[k] = (n_k**2)/((n_k**2)+2)
            similarity_matrix[i][j] = np.sum(Sx_y)/d
    print("Please be a real thing: ", similarity_matrix)
    return similarity_matrix

# testy test
# listylist = [[1,2],[2,3],[4,5]]
# categorical_preprocessing_csv(listylist)