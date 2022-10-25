import Multilayer_Perceptron 
from Multilayer_Perceptron import Perceptron
from Multilayer_Perceptron import MultiLayerPerceptron
#This class tests the perceptron class as logic gates

#This test simulates an "AND" gate using perceptrons
print("\nTraining Neural Network as an AND Gate:\n")
neuron = Perceptron(inputs=2)
neuron.set_weights([10,10,-15]) #AND

#will always return same answer since weights are constant
#last term (3rd) is not included in number of inputs, represents bias

print("OR Gate:")
print ("0 0 = {0:.10f}".format(neuron.run([0,0])))
print ("0 1 = {0:.10f}".format(neuron.run([0,1])))
print ("1 0 = {0:.10f}".format(neuron.run([1,0])))
print ("1 1 = {0:.10f}".format(neuron.run([1,1])))


#This test simulates an "OR" gate using perceptrons'
print("\nTraining Neural Network as an OR Gate:\n")
neuron = Perceptron(inputs=2)
neuron.set_weights([15,15,-10]) #OR

print("AND Gate:")
print ("0 0 = {0:.10f}".format(neuron.run([0,0])))
print ("0 1 = {0:.10f}".format(neuron.run([0,1])))
print ("1 0 = {0:.10f}".format(neuron.run([1,0])))
print ("1 1 = {0:.10f}".format(neuron.run([1,1])))


#This test simulates an "XOR" gate using perceptrons
#Involves attaching mutliple OR and AND gates together
print("\nTraining Neural Network as an XOR Gate:\n")
mlp = MultiLayerPerceptron(layers=[2,2,1])  #mlp
mlp.set_weights([[[-10,-10,15],[15,15,-10]],[[10,10,-15]]])
mlp.printWeights()
print("MLP:")
print ("0 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print ("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print ("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print ("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))