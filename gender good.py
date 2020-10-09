# -*- coding: utf-8 -*-
"""
Maayan Eliya, Shir Gavriel, Amit Caspi, Roni Peri, Ron Amado, Topaz Ben Atar, Maya Yosef, Hila Daniel

The script below predict someone's gender given their weight and height. 
We will represent male with a 0 and female with a 1.
"""

import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training


class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer
    - an output layer with 1 neuron

  '''
  
  def __init__(self, inputs, outputs):
    '''
      ----------
      inputs : inputs = data (np.array), true_outputs = all_y_trues (np.array)
      outputs : constructive function
      -------
     '''
     
    hiddenLayerNeurons, outputLayerNeurons = 2,1 #the amount of neurons in the hidden and output layers
    self.inputs  = inputs.astype(float) #inputs consist of weight and height
    self.true_outputs = outputs #the original output
    inputLayerNeurons = np.shape(self.inputs)[1] #the amount of neurons in the input layer
      # Weights
    self.hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons)) #the initial hidden weights
    self.output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons)) #the initial output weights
       # Biases
    self.hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons)) #the initial hidden biases
    self.output_bias = np.random.uniform(size=(1,outputLayerNeurons)) #the initial output biases
    
    self.error_history = [] #a list consist of the loss values
    self.epoch_list = [] #a list consist of the epochs
    
    print("Initial hidden weights: ",end='')
    print(self.hidden_weights)
    print("Initial hidden biases: ",end='')
    print(self.hidden_bias)
    print("Initial output weights: ",end='')
    print(self.output_weights)
    print("Initial output biases: ",end='')
    print(self.output_bias)
        
        
  def sigmoid (self, x):
   # Sigmoid activation function: f(x) = 1 / (1 + e^(-x)). Return the result.
     return 1 / (1 + np.exp(-x))
  
    
  def deriv_sigmoid(self, x):
   #Derivative of sigmoid: f'(x) = f(x) * (1 - f(x)). Return the result.
    fx = self.sigmoid(x)
    return fx * (1 - fx)
  
    
  def mse_loss(self, error):
    #mean squared error. Return the result.
    return np.average(np.abs(error))
               

  def feedforward(self):
    '''The function returns hidden and predicted_output'''
     
    # do dot of hidden_weights and inputs, add hidden_bias, then use the sigmoid activation function
    hidden = np.dot(self.inputs, self.hidden_weights) #output of the hidden layer and the input to the output layer
    hidden += self.hidden_bias
    hidden = self.sigmoid(hidden)
    
    # do dot of output_weights and hidden, add output_bias, then use the sigmoid activation function
    predicted_output = np.dot(hidden,self.output_weights) #our neural network output
    predicted_output += self.output_bias
    predicted_output = self.sigmoid(predicted_output)
    return (predicted_output,hidden)


  def backpropagation (self, predicted_output, hidden_layer_output ,lr):
    '''
      ----------
      inputs : predicted_output, hidden_layer_output, learning rate
      outputs : calculate error and return it, and do the Backpropagation progress - 
      the function calculate partial derivatives by working backwards.
      -------
     ''' 
     
    error = self.true_outputs - predicted_output #the original output minus our neural network output
    d_output = error * self.deriv_sigmoid(predicted_output) # the output's derivation
        	
    error_hidden_layer = d_output.dot(self.output_weights.T) #calculate the hidden layer's error by doing dot of d_output and output_weights 
    d_hidden_layer = error_hidden_layer * self.deriv_sigmoid(hidden_layer_output) # the hidden's derivation
        
    #Updating Weights and Biases
    self.output_weights += hidden_layer_output.T.dot(d_output) * lr
    self.output_bias += np.sum(d_output,axis=0,keepdims=True) * lr
    self.hidden_weights += self.inputs.T.dot(d_hidden_layer) * lr
    self.hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr
    return error
  
    
  def train(self,epochs=10000): #epochs= number of times to loop through the entire dataset

    '''
    - self.inputs is a [n x number of features] numpy array, n = number of samples in the inputs array.
      Elements in self.true_outputs correspond to those in self.inputs.
    '''
    learn_rate = 0.1 #learning rate = a constant that controls how fast the train change, moving toward a minimum of a loss function.

    for epoch in range (epochs):
        # --- Do a feedforward
        (predicted_output, hidden_layer_output) =  self.feedforward() #do the feedforward progress

        error=self.backpropagation (predicted_output, hidden_layer_output, learn_rate) # calculate error and calculate partial derivatives in backpropagation.

      # keep track of the error history over each epoch
        self.error_history.append(self.mse_loss(error))
        self.epoch_list.append(epoch)
        
    print("Final hidden weights: ",end='')
    print(self.hidden_weights)
    print("Final hidden bias: ",end='')
    print(self.hidden_bias)
    print("Final output weights: ",end='')
    print(self.output_weights)
    print("Final output bias: ",end='')
    print(self.output_bias)
    print("\nOutput from neural network after epochs: ",end='')
    print(predicted_output)

    
    
def main():
    ##

    # Define dataset
    data = np.array([
      [-2, -1],  # Alice
      [25, 6],   # Bob
      [17, 4],   # Charlie
      [-15, -6], # Diana
    ]) ##inputs #[weight, height]
    #weight = real weight-135
   
    all_y_trues = np.array([
      [1], # Alice
      [0], # Bob
      [0], # Charlie
      [1], # Diana
    ]) #the original output
    
    # Train our neural network
    network = OurNeuralNetwork(data,all_y_trues) #network = an object of OurNeuralNetwork class
    epochs=10000 #epochs= number of times to loop through the entire dataset
    network.train(epochs) #do the train
    
   # Do a graph - plot the error over the entire training duration
    plt.figure(figsize=(15,5)) #defines the graph size
    plt.plot(network.epoch_list, network.error_history) #plot network.epoch_list (x) and network.error_history (y)
    plt.xlabel('Epoch') #the x axis's name = Epoch
    plt.ylabel('Loss') #the y axis's name = Loss
    plt.show()


if __name__ == "__main__": 
    main()
    