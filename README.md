# General info
This is an implementation of a Feedforward Neural Network created using Python. The project also contains training data to utilize the network as a Segment Display Classifier, essentially taking in the brightness levels of a seven-segment display and returning the most probable number associated with that combination. In particular, the training data makes use of one-hot encoding.


## Files
- **Multilayer_Perceptron**: Provides structure for the network, and each individual neuron itself.
- **MLP_Logic_Gate_Tester**: A test class that demonstrates how perceptrons can be used as logic gates.
- **SDR_Sample_Data.txt**: The training data used by both the terminal and graphical SDR neural networks.
- **Terminal_SDR_Neural_Network**: A terminal-based segment display classifier that takes in an array representing segment brightness levels and returns similar matches.
- **SDR_Test_Queries.txt**: Test data to be used by the terminal SDR neural network.
- **Graphical_SDR_Neural_Network**: A graphic-based segment display classifier. The training data is in *SDR_Sample_Data.txt*, just like in the terminal version, but this version uses Tkinter-based GUI to receive and output test data.
