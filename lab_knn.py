import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


iris = sb.load_dataset('iris')
irisX = iris.iloc[:,:-1]
irisY = iris.iloc[:,4]
print(irisX)
print(irisY)
