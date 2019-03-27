import numpy as np 
import pandas as pd
from numpy.linalg import inv
from numpy import matmul as mm
from numpy import transpose as tp
from numpy.linalg import eigvals

import difflib

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold



#model creation/evaluation functions
# 
# 
# 
# 
def calculate_w(reg, x, y):
  d = x.shape[1]
  covar = mm(tp(x),x )
  lambdai = np.diag( np.ones(d)*reg  )
  addedmatrix = lambdai + covar
  inverse = inv(addedmatrix)

  rightside = mm(tp(x), y) 

  w = mm(inverse,rightside)
  return w

def calculate_mse(w, x, y):
  n = x.shape[0]

  postweights =  mm(x,w)
  
  errorvector = postweights - y
  squarederror = np.square(errorvector)

  mse = np.mean(squarederror)
  
  return mse

# 
# 
# 
# 
# 
# 
# 
# end of functions




datafiles = [
  "test-100-10.csv",
  "test-100-100.csv",
  "test-1000-100.csv",
  "test-forestfire.csv",
  "test-realestate.csv",
  "testR-100-10.csv",
  "testR-100-100.csv",
  "testR-1000-100.csv",
  "testR-forestfire.csv",
  "testR-realestate.csv",
  "train-100-10.csv",
  "train-100-100.csv",
  "train-1000-100.csv",
  "train-forestfire.csv",
  "train-realestate.csv",
  "trainR-100-10.csv",
  "trainR-100-100.csv",
  "trainR-1000-100.csv",
  "trainR-forestfire.csv",
  "trainR-realestate.csv"
]




def csv_values(filename):
  df= pd.read_csv(filename)
  #returns a dataframe...taken care of later when constructing datasets
  return df


#construct preliminary datadict
datadict = {}
for filename in datafiles:
  datadict[filename] = csv_values(filename)


# creates the list of target file names
yfiles = []
for filename in datafiles:
  if ("R" in filename):
    datafiles.remove(filename)
    yfiles.append(filename)

for filename in datafiles:
  if ("R" in filename):
    datafiles.remove(filename)
    yfiles.append(filename)

for filename in datafiles:
  if ("R" in filename):
    datafiles.remove(filename)
    yfiles.append(filename)

#literally have no idea why looping 3 times is necessary but it is


datasets = {}
#given a filename, find its match, determine if test or train, then create dict
#key is training file name.  
for filename in datafiles:
  training = None
  test = None
  if ("test" in filename):
    test = filename
    training = filename.replace("test","train")
  else:
    training = filename
    test = filename.replace("train", "test")



  trainingy = difflib.get_close_matches(training, yfiles, 1 )[0] #this is the corresponding y file
  testy = difflib.get_close_matches(test, yfiles, 1 )[0] #this is the corresponding y file


  dataname = (training[:-4]).replace("train-","")
  dataset = {}
  trainingpair = {}
  testpair = {}
  trainingpair["x"] = datadict[training].values #add respective values to dict
  trainingpair["y"] = np.array( list( map(lambda x : x[0], datadict[trainingy].values) )  )


  testpair["x"] = datadict[test].values
  testpair["y"] = datadict[testy].values
  dataset["train"] = trainingpair
  dataset["test"] = testpair
  datasets[dataname] = dataset

#e.g..  datasets["1000-100"]["train"]["x"] returns training x data for 1000-100 dataset


#construct the new smaller datasets
fifty = {}
fifty["train"] = {}
fifty["test"] = {}
fifty["train"]["x"] = datasets["1000-100"]["train"]["x"][0:50]
fifty["train"]["y"] = datasets["1000-100"]["train"]["y"][0:50]
fifty["test"]["x"] = datasets["1000-100"]["test"]["x"]
fifty["test"]["y"] = datasets["1000-100"]["test"]["y"]
datasets["50(1000)-100"] = fifty

hundo = {}
hundo["train"] = {}
hundo["test"] = {}
hundo["train"]["x"] = datasets["1000-100"]["train"]["x"][0:100]
hundo["train"]["y"] = datasets["1000-100"]["train"]["y"][0:100]
hundo["test"]["x"] = datasets["1000-100"]["test"]["x"]
hundo["test"]["y"] = datasets["1000-100"]["test"]["y"]
datasets["100(1000)-100"] = hundo

onefif = {}
onefif["train"] = {}
onefif["test"] = {}
onefif["train"]["x"] = datasets["1000-100"]["train"]["x"][0:150]
onefif["train"]["y"] = datasets["1000-100"]["train"]["y"][0:150]
onefif["test"]["x"] = datasets["1000-100"]["test"]["x"]
onefif["test"]["y"] = datasets["1000-100"]["test"]["y"]
datasets["150(1000)-100"] = onefif





#Experimenting with regularizers
#Over a range of regularizers from 10-150, find train and test MSE for each dataset and plot
####
#
#
#

for dataset in datasets:
  trainMSEs = []
  testMSEs = []
  for reg in range(0,150,10):

    trainx = datasets[dataset]["train"]["x"]
    trainy = datasets[dataset]["train"]["y"]
    testx = datasets[dataset]["test"]["x"]
    testy = datasets[dataset]["test"]["y"]

    w = calculate_w(reg, trainx, trainy)

    trainMSE = calculate_mse(w, trainx, trainy)
    testMSE = calculate_mse(w, testx, testy)
    trainMSEs.append(trainMSE)
    testMSEs.append(testMSE)


    print("Dataset is: " + dataset + "with reg:  " + str(reg))
    print("\ttrainMSE is: " + str(trainMSE))
    print("\ttestMSE is: " + str(testMSE))
  #out of for loop have all MSEs for each reg
  # plt.figure(1)
  regs = list(range(0,150,10))

  #CODE FOR PLOTTING
  #edge case of unrealistically high mses at reg = 0 for these datasets
  # if (dataset == "100-100" or dataset == "50(1000)-100" or dataset == "100(1000)-100" or dataset == "150(1000)-100"):
  #   testMSEs[0] = 10
  #   trainMSEs[0] = 0

  # plt.plot(regs, trainMSEs, label = "trainMSE")
  # plt.plot(regs, testMSEs, label = "testMSE")
  # plt.legend()
  # plt.xlabel('regularizer')
  # plt.ylabel('MSE')
  # plt.title("train /test set MSE vs regularizer for: " + dataset)
  # plt.show()	

#
#
#
#
#
#
#END OF TASK 1

    







#CROSS VALIDATION TO PICK BEST LAMBDA FOR EVERY DATASET AND THEN USE TO CALC TEST MSE 
#
#
#
#
#
#

def find_best_reg(x,y, ksplits = 10):
  kf = KFold(n_splits=ksplits)
  
  #initialize mse sum dict
  msesumdict = {}
  for reg in range(0,150,10):
    msesumdict[str(reg)] = 0

  #for every k fold split of the data
  for trainindex,testindex in kf.split(x):
    trainx, trainy = x[trainindex], y[trainindex]
    valx, valy = x[testindex], y[testindex]
    #for every possible value of the regularizer, calculate test MSE and add to running sum for particular reg value
    for reg in range(0,150,10):
      w = calculate_w(reg,trainx,trainy)
      valMSE = calculate_mse(w, valx, valy)
      msesumdict[str(reg)] += valMSE

  #after all running sums are calculated find the minimum
  minreg = int( min(msesumdict, key= msesumdict.get) )
  return minreg


print("FINDING TEST SET MSE FOR CROSS VALIDATED BEST REGULARIZER")

#for every dataset find best regularizer and compute test MSE
for dataset in datasets:
  trainx = datasets[dataset]["train"]["x"]
  trainy = datasets[dataset]["train"]["y"]
  testx = datasets[dataset]["test"]["x"]
  testy = datasets[dataset]["test"]["y"]
  reg = find_best_reg(trainx, trainy)
  w = calculate_w(reg, trainx, trainy)
  testMSE = calculate_mse(w,testx, testy)
  print("Dataset is: " + dataset + " with cross validated best reg:  " + str(reg))
  print("\ttestMSE is: " + str(testMSE))



#
#
#
#
#End of cross validation













#BAYESIAN FUNCTIONS 
#
#
#
#
#

def mun(beta, sn, x, y):

  rightmost = mm(tp(x), y)
  mn = beta * mm(sn,rightmost)
  return mn



def sn(alpha, beta, x):
  d = x.shape[1]
  alphamat = np.diag( np.ones(d)*alpha  )
  betamat = beta * mm(tp(x),x)

  add = alphamat + betamat
  sn = inv(add)
  return sn

def squiggle(alpha, beta, x):
  lambdas = eigvals(  beta*mm(tp(x),x)  )
  denom = alpha + lambdas
  quotient = np.divide(lambdas,denom)
  summed =  np.sum(quotient)
  return summed


def newalpha(squig,mn):
  dotted = mm( tp(mn) ,mn)

  return squig/dotted


def newbeta(squig, x, y, mn):
  n = x.shape[0]
  mse = calculate_mse(mn, x, y)
  leftside = n/(n - squig)
  return leftside*mse



def optimalbayesian(x, y, alpha = 1,beta = 1, countmax = 1000, threshold = 0.0001 ):
  prevalpha = alpha
  prevbeta = beta
  count = 0
  updating = True
  while(updating):
 
    currsn = sn(alpha, beta ,x)
    currmn = mun(beta, currsn, x, y)

    squig = squiggle(alpha, beta, x)
    prevalpha = alpha
    prevbeta = beta
    alpha = newalpha(squig, currmn)

    beta = newbeta(squig, x, y, currmn)

    if ( np.abs(prevalpha - alpha) < threshold or count > countmax ):
      weights = currmn
      return alpha, beta, weights

    count+=1




print(":::BAYESIAN MODELS ONLY:::")
for dataset in datasets:
  print("\t" + dataset)
  trainingx = datasets[dataset]["train"]["x"]
  trainingy = datasets[dataset]["train"]["y"]
  testx = datasets[dataset]["test"]["x"]
  testy = datasets[dataset]["test"]["y"]
  alpha, beta, weights = optimalbayesian(trainingx, trainingy)

  trainMSE = calculate_mse(weights, trainingx, trainingy)
  testMSE = calculate_mse(weights, testx, testy)
  datasets[dataset]["train"]["bayesianMSE"] = trainMSE
  datasets[dataset]["test"]["bayesianMSE"] = testMSE
  print("\ttrainMSE: " + str(trainMSE))
  print("\ttestMSE: " + str(testMSE))


#
#
#
#
#
#
#END OF BAYESIAN















