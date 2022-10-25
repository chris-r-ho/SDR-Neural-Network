from operator import concat
import Multilayer_Perceptron 
from Multilayer_Perceptron import MultiLayerPerceptron
import numpy as np

#This program implements one-hot encoding to classify SDRs into numbers

#File to access training data
samplesFileName = 'SDR_Sample_Data.txt' 

#File to access queries
queriesFileName = 'SDR_Test_Queries.txt' 

#Show the values of each bin for every query
showAllBins = True 

#If all values are shown, sort them from largest bin value to smallest
sorted = True 

#number of epochs/iterations to train the neural network over
epochs = 3000

#interval of number of epochs to print the current training error
printInterval = 100

print("\nSegment Display Recognition (SDR) Neural Network Implementation\n")

file = open(samplesFileName, 'r')


#read in network (neuron distribution per layer) from first line of sample data
#each subsequent line represents a pair of training inputs and outputs
network = [int(y) for y in np.array([np.array(x) for x in file.readline().strip().split(',')],dtype=object)]
mlp = MultiLayerPerceptron(layers=network)



print("Preset distribution of neurons in each network layer:",network,"\n")

#Read training data inputs and outputs
numSets = 0 #Training set counter
t_inputs = [] #Program training data inputs
t_outputs = [] #Expected training data outputs
binNames = [] #Name of each output bin

print("Phase 1: Feeding in sample data into the network\n")

#read in training data, as well as bin labels
while True:
    line = file.readline()
    if not line:
        break
    pair = line.strip().split(' ')
    binName = concat('Bin #',str(numSets))
    if len(pair)>2:
        binName = ''
        for word in pair[2:]:
            binName+=(word+" ")

    #Add each training data set
    t_inputs.append([int(x) for x in pair[0].split(',')])
    t_outputs.append([int(x) for x in pair[1].split(',')])
    binNames.append(binName.strip())
    #output info about each training set as they are read in
    print("Loading training set {} - Bin Name: \'{}\' Input layer: {} Corresponding output layer: {}".format(numSets, binNames[numSets],t_inputs[numSets],t_outputs[numSets]))
    numSets += 1

t_outputs = np.array([np.array(x) for x in t_outputs],dtype=object)
t_inputs = np.array([np.array(x) for x in t_inputs],dtype=object)


file.close()

print("\nSample data loading complete.\n\nPhase 1 is complete.\n\n")

print("\nPhase 2: Training the network\n")
#test code


print("Training program for",epochs,"iterations ('epochs'), printing updates every",printInterval,"iterations:\n")

#The backpropagation algorithm is continually invoked to reduce the mean square error
for i in range(epochs):
    MSE = 0.0
    for j in range(numSets):
        MSE += mlp.bp(t_inputs[j],t_outputs[j])    

    MSE /= 10
    if(i%printInterval == 0):
        print ("Epoch",i,"/",epochs,"complete. Current Training Error:",MSE)
print("\nNetwork successfully trained over",epochs,"epochs.\n\nPhase 2 is complete.\n\n")

#print("Weight of each neuron conection in each layer:")
#mlp.printWeights()
    
print("Phase 3: Testing the network with custom queries\n\n")

file = open(queriesFileName, 'r')


query = 1
while True:
    #read in each query
    line = file.readline()
    if not line:
        break
    input = [int(x) for x in line.split(',')]

    #store all the bin values for a given query
    binValues = {}
    for i in range(len(binNames)):
        binValues[i]=mlp.run(input)[i]

    #display the most applicable bin
    j = mlp.largestBin(input)
    print("\nQuery",query,": ",input, "most closely corresponds to",binNames[j],"(Bin index "+str(j)+") with a certainty of "+str(binValues[j]))
    
    if showAllBins:
        if sorted:
            print("All bin values for this query, sorted from least to greatest index:")
            for i in range(len(binNames)):
                next = max(binValues, key=binValues.get)
                print(str(binNames[next])+", bin at index",next,":",binValues[next])
                binValues.pop(next)
        else:
            print("All bin values for this query, sorted from highest to lowest similarity:")
            for i in range(len(binNames)):
                print(str(binNames[i])+", bin at index",i,":",binValues[i])
    query+=1

file.close()

print("\nPhase 3 is complete.\n\nThe neural network has successfully processed all queries.\n\n")