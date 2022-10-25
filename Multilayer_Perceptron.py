import numpy as np

class Perceptron:
    #A single neuron with the sigmoid activation function.=
    
    #inputs: The number of inputs in the perceptron, not counting the bias
    #bias:   The bias term. By default it's 1.0

    def __init__(self, inputs, bias = 1.0):
        #Return a new Perceptron object with the specified number of inputs (+1 for the bias)

        #Weights will be an array (length inputs+1) of random floating point numbers between -1 and +1
        self.weights = (np.random.rand(inputs+1)*2)-1
        self.bias = bias

    def run(self, x):
        #Run the perceptron. x is a python list with the input values.=
        #Dot product between the (inputs & bias) and the weights

        sum = np.dot(np.append(x,self.bias),self.weights)

        #sigmoid function ensures that outputs are between 0 and 1
        return self.sigmoid(sum)
        
    def set_weights(self, w_init):
        # Assign array version of arguments to class weight
        self.weights = np.array(w_init)
       

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        # return the output of the sigmoid function applied to x


#Contains layers of perceptron classes
class MultiLayerPerceptron:     
    # multilayer perceptron class that uses the Perceptron class above

        #layers:  A python list with the number of elements per layer
        #bias:    The bias term. The same bias is used for all neurons
        #eta:     The learning rate

    def __init__(self, layers, bias = 1.0, eta=0.5):
        #Return a new MLP object with the specified parameters

        #number of neurons per layer, including empty input layer
        self.layers = np.array(layers,dtype=object)

        self.bias = bias
        self.eta = eta
        #numpy array of numpy arrays
        self.network = [] # The list of lists of neurons

        #holds output values
        self.values = []  # The list of lists of output values        
        self.d = []       # The list of lists of error terms (lowercase deltas)
        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]#inner loop sets values equal to 0
            self.d[i] = [0.0 for j in range(self.layers[i])]#set initial errors equal to d
            if i>0: #network[0] is input layer, does not have any neurons
                for k in range(self.layers[i]):
                    #Add new perceptrons to the network, connecting to all previous perceptrons
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
        
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
        self.d = np.array([np.array(x) for x in self.d],dtype=object)
    
    def set_weights(self, w_init):
        #set weights for all but the initial layer
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])#nothing in first layer, indexing at 1+i

    def printWeights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
        print()

    def run(self, x):
        #Feed a sample x into the MultiLayer Perceptron
        x = np.array(x,dtype=object)
        self.values[0] = x
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):  
                #compute the value of each neuron based on all its connections to the previous layer
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        #Return the last layer
        return self.values[-1]




    def bp(self, x, y):
        #Run a single (x,y) pair with the backpropagation algorithm
        x = np.array(x,dtype=object) #inputs
        y = np.array(y,dtype=object) #outputs

        # Backpropagation steps

        # STEP 1: Feed a sample to the network 
        outputs = self.run(x)
        
        # STEP 2: Calculate the MSE
        error = (y - outputs)

        #Divide sum of squares of errors by number of neurons in last layer
        MSE = sum( error ** 2) / self.layers[-1]

        # STEP 3: Calculate the output error terms
        #deltas of last row
        self.d[-1] = outputs * (1 - outputs) * (error)

        # STEP 4: Calculate the error term of each unit on each layer

        #From second to last layer to first layer
        for i in reversed(range(1,len(self.network)-1)):
            #for every neuron in this layer
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                #for our one neuron - connections to all neurons in layer in front of current layer
                for k in range(self.layers[i+1]): 
                    #multiply connected neuron error * connected neuron's weight for our current neuron
                    fwd_error += self.network[i+1][k].weights[h] * self.d[i+1][k]  
                #error term for this neuron, using the previous fwd error result             
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error

        # STEPS 5 & 6: Calculate the deltas and update the weights
        #For each layer in the netowrk
        for i in range(1,len(self.network)):
            #For each neuron in the layer
            for j in range(self.layers[i]):
                # For each of our neuron's connections to the previous layer + 1 for bias term
                for k in range(self.layers[i-1]+1):
                    #Delta will be based on biased term if the last in the layer
                    if k==self.layers[i-1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    #otherwise, it will be based on the weight of the output value
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
        return MSE

    def largestBin(self, input):
        #returns the index of the largest output bin
        bins = self.run(np.array(input,dtype=object))
        max = int(bins[0]);   
        self.run(input)
        for i in range(len(bins)):    
            if bins[i]>bins[max]:
                max = i
        return max
  
        


