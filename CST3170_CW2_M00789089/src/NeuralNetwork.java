import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * This class is an artificial neural network that performs digit classification.
 * It is a fully connected feed forward network that is trained through back propagation of the error 
 * calculated at the output layer.
 * ReLU is used as the activation function for the hidden layers and softmax is used for the output layer.
 * The error is calculated using the cross entropy loss function.
 * The optimiser used for updating weights and biases is stochastic gradient descent.
 * 
 * @version 1.0 22/02/2024
 * @author M00789089
 * 
 */
public class NeuralNetwork {
	
    private Layer[] layers;			// Array to store layer objects.
    private int epochs;				// Number of epochs to train model.
    private double learningRate;	// Learning rate for stochastic gradient descent.
    private double dropoutRate;		// Dropout rate for regularization.
    private int[] layerSizes;		// Array for the sizes of layers.

    /**
     * Constructor used to build the neural network and create layers.
     * 
     * @param inputLayerSize	Number of data points in an input sample .
     * @param layerSizes		Array of layer sizes.
     * @param epochs			Number of epochs to train the model.
     * @param learningRate		Learning rate value.
     * @param dropoutRate		Dropout rate value.
     */
    public NeuralNetwork(int inputLayerSize, int[] layerSizes, int epochs, double learningRate, double dropoutRate) {
    	this.epochs = epochs;
    	this.learningRate = learningRate;
    	this.dropoutRate = dropoutRate;
        this.layerSizes = layerSizes;
        this.layers = new Layer[layerSizes.length];

        /* Creating layer objects with their respective sizes. */
        for (int layerPos = 0; layerPos < layerSizes.length; layerPos++) {
            /* The first hidden layer takes the size of the input sample instead of the 
             * size of previous layer as the number of input neurons to the layer. */
        	if (layerPos == 0) {
            	layers[layerPos] = new Layer(layerSizes[layerPos], inputLayerSize);	
            } else {	
                layers[layerPos] = new Layer(layerSizes[layerPos], layerSizes[layerPos - 1]);
            }
        }
    }
    
    /**
     * This method trains the model on the training set through forward and
     * backward propagation over a number of epochs.
     * 
     * @param trainingSet	List of object arrays representing digits in the training set.
     */
    public void train(List<Object[]> trainingSet) {
    	int trainingSetSize = trainingSet.size();							// Get training set size.
        int outputLayerSize = this.layerSizes[this.layerSizes.length - 1];	// Get output layer size.
        
        /* Flag set to true for training. 
         * Used to apply dropout during training only. */
        boolean useDropout = true;
        
        /* Train the model over specified number of epochs. */
        for (int count = 0; count < this.epochs; count++) {
            
        	/* Shuffling the dataset to prevent the model from learning the order of the input samples. */
        	Collections.shuffle(trainingSet);
        	
            double totalLoss = 0.0;	// Initialise total loss for the epoch.
            
            /* Feed each of the input samples from the training set into the model,
             * back propagate, and update the weights and biases. */
            for (Object[] digit : trainingSet) {
                double[] inputValues = (double[]) digit[0];		// Store data points of input sample in array.
                int label = (int) digit[1];						// Store label of input sample.
                
                /* Store the activation values of output layer neurons after feeding forward. */
                double[] outputValues = feedForward(inputValues, outputLayerSize, useDropout);
                
                /* Accumulate the loss calculated with cross-entropy for each pass. */
                totalLoss += calculateCrossEntropyLoss(outputValues, label);
                
                resetGradients();	// Reset gradient values.
                propagateBackwards(inputValues, outputValues, label, outputLayerSize);	// Perform backwards propagation.
                updateParameters();		// Update all weights and biases in the network.
            }
            
            /* Calculate and print out the average loss for the epoch. */
            double averageLoss = totalLoss / trainingSetSize;
            System.out.println("Average Loss for Epoch " + (count + 1) + ": " + averageLoss);
        }
    }

    /**
     * This method feeds a testing set into the network, gets the predicted class from the output layer
     * and compares it to the actual label for each input. The accuracy is calculated based on the number
     * of correct predictions.
     * 
     * @param testingSet	List of object arrays representing digits in the testing set.
     * @return	The accuracy as a value of type double.
     */
    public double evaluate(List<Object[]> testingSet) {
        int outputLayerSize = this.layerSizes[this.layerSizes.length - 1];	// Store output layer size.
        
        /* Set flag to false to prevent neurons from being dropped. */
        boolean useDropout = false;
        
        int correctPredictions = 0;	// Initialise number of correct predictions as zero.
        
        /* Go through the testing set and feed each input sample into the model. */
        for (Object[] digit : testingSet) {
        	
        	/* Store data points and label of input sample. */
            double[] inputValues = (double[]) digit[0];
            int actualLabel = (int) digit[1];
            
            /* Feed data points into model and get predicted digit. */
            double[] outputValues = feedForward(inputValues, outputLayerSize, useDropout);
            int predictedLabel = getPrediction(outputValues);
            
            /* Increment correct predictions when predicted label matches actual label. */
            if (actualLabel == predictedLabel) {
                correctPredictions++;
            }
        }
        
        /* Calculate accuracy as a percentage. */
        double accuracy = ((double) correctPredictions / testingSet.size()) * 100;
        return accuracy;
    }

    /**
     * Performs forward pass of the input values through each of the layers.
     * Calculates the weighted sums and activations of neurons in each layer.
     * Regularization is performed by randomly dropping neurons from the hidden layers.
     * 
     * @param inputValues		Array of data points of an input sample.
     * @param outputLayerSize	Size of the output layer.
     * @param useDropout		Flag to check if neurons should be dropped.
     * @return	Array storing activation values of neurons in the output layer.
     */
    private double[] feedForward(double[] inputValues, int outputLayerSize, boolean useDropout) {
    	
    	/* Initialise empty array to store activations of output layer neurons. */
    	double[] outputValues = new double[outputLayerSize];
    	
    	boolean isOutputLayer;	// Flag to check whether a layer is the output layer.
    	
    	/* Go through each layer in the network */
    	for (int layerPos = 0; layerPos < this.layers.length; layerPos++) {
    		
    		/* Use data points from input sample as input for the first hidden layer
    		 * to calculate weighted sums and activations of neurons. Layers other than the
    		 * first hidden layer take neuron activations from the previous layer in the network
    		 * as input for weighted sum calculation. */
    		if (layerPos == 0) {
    			isOutputLayer = false;
    			layers[layerPos].calculateWeightedSums(inputValues);
    			calculateActivations(layers[layerPos], isOutputLayer, useDropout);	
    		} else {
    			layers[layerPos].calculateWeightedSums(layers[layerPos - 1].getActivations());
    			
    			/* Checking whether layer is output layer or not. The last layer in the array is the output layer. */
    			if (layerPos == (this.layers.length - 1)) {
    				isOutputLayer = true;	// Set flag to true for output layer
    				calculateActivations(layers[layerPos], isOutputLayer, useDropout);
    				outputValues = layers[layerPos].getActivations();	// Store output layer activation values.
    			} else {
    				isOutputLayer = false;
    				calculateActivations(layers[layerPos], isOutputLayer, useDropout);	
    			}
    		}
    	}
    	return outputValues;
    }
    
    /**
     * Backward propagation is based on the chain rule, the error is propagated back through the network 
     * to determine by how much each neuron and the weights between neurons contributed to the error.
     * The gradients of loss at the output layer is calculated using the cross entropy loss function.
     * Calculates gradients of loss with respect to weights/biases.
     * Stores the gradients of loss for weights and biases in each layer.
     *
     * @param inputValues       Array of data points of the input sample fed into the network.
     * @param outputValues      Array of activation values of output layer neurons.
     * @param digitLabel        Actual label of input sample.
     * @param outputLayerSize   Size of the output layer.
     */
    private void propagateBackwards(double[] inputValues, double[] outputValues, int digitLabel, int outputLayerSize) {
    	
    	/* Encode digit label using one-hot encoding. */
        double[] labelOneHot = new double[outputLayerSize];
        labelOneHot[digitLabel] = 1.0;
        
        double[] errorGradients = new double[outputLayerSize];	// Empty array to store error gradients calculated at output layer.
        
        /* Calculate the gradient of loss with respect to output 
         * layer activations using cross-entropy loss function. */
        for (int i = 0; i < outputValues.length; i++) {
        	errorGradients[i] = outputValues[i] - labelOneHot[i];
        }
        
        /* Start from output layer and back propagate error through the network. */
        for (int layerPos = (layers.length - 1); layerPos >= 0; layerPos--) {
            Layer currentLayer = layers[layerPos];	// Get the layer object at the current position in the array of layers.
            
            /* The inputs to a neuron in the current layer are the activation values of neurons from the previous layer, 
             * except for the first hidden layer which takes the digit pixel values as input. */
            double[] inputs = null;
            if (layerPos == 0) {
                inputs = inputValues;
            } else {
                inputs = layers[layerPos - 1].getActivations();
            }
            
            /* Initialise temporary arrays to store calculated gradients for weights and biases. */
            double[][] tempWeightGradients = new double[currentLayer.getActivations().length][inputs.length];
            double[] tempBiasGradients = new double[currentLayer.getActivations().length];
            
            /* Go through each neuron in the layer and calculate error gradients with respect to weights and bias. */
            for (int neuronIndex = 0; neuronIndex < currentLayer.getActivations().length; neuronIndex++) {
                for (int connectionIndex = 0; connectionIndex < currentLayer.getWeights()[neuronIndex].length; connectionIndex++) {
                    tempWeightGradients[neuronIndex][connectionIndex] = errorGradients[neuronIndex] * inputs[connectionIndex];
                }
                tempBiasGradients[neuronIndex] = errorGradients[neuronIndex];
            }
            
            /* Store gradients for current layer. */
            currentLayer.setWeightGradients(tempWeightGradients);
            currentLayer.setBiasGradients(tempBiasGradients);
            
            /* Calculate gradient of loss with respect to activations from previous layer. */
            if (layerPos > 0) {
            	
            	/* Initialise empty array to store error gradients for previous layer. */
                double[] previousErrorGradients = new double[layers[layerPos - 1].getActivations().length];
                
                /* Calculate error gradients with respect to activations for previous layer. */
                for (int previousNeuron = 0; previousNeuron < layers[layerPos - 1].getActivations().length; previousNeuron++) {
                	previousErrorGradients[previousNeuron] = 0;
                    for (int currentNeuron = 0; currentNeuron < currentLayer.getActivations().length; currentNeuron++) {
                    	previousErrorGradients[previousNeuron] += errorGradients[currentNeuron] * currentLayer.getWeights()[currentNeuron][previousNeuron];
                    }
                    
                    /* Apply derivative of ReLu activation function to calculated gradients. */
                    previousErrorGradients[previousNeuron] *= getReluDerivative(layers[layerPos - 1].getActivations()[previousNeuron]);
                }
                
                /* Set error gradients for the next iteration (previous layer). */
                errorGradients = previousErrorGradients;
            }
        }
    }
    
    /**
     * This method calculates the activations of neurons in the layers from their weighted sums 
     * and carries out regularization by randomly dropping neurons. The activation functions 
     * used are ReLu for the hidden layers and softmax for the output layer.
     * 
     * @param layer				Layer object.
     * @param outputLayerCheck	Flag to check if layer is output layer.
     * @param useDropout		Flag to check whether neuron drop out should be performed.
     */
    private void calculateActivations(Layer layer, boolean outputLayerCheck, boolean useDropout) {
        
    	/* Apply ReLu or softmax to weighted sums depending on which layer is being processed. */
    	if (outputLayerCheck) {
            layer.setActivations(applySoftmax(layer.getWeightedSums()));
        } else {
            double[] weightedSums = layer.getWeightedSums();				// Get weighted sums of neurons in layer.
            double[] activations = new double[weightedSums.length];			// Initialise empty array to store calculated activations.
            
            /* Calculate activations by applying ReLu to each weighted sum. */
            for (int neuronIndex = 0; neuronIndex < weightedSums.length; neuronIndex++) {
                activations[neuronIndex] = applyRelu(weightedSums[neuronIndex]);
            }
            
            /* Drop neurons if needed. */
            if (useDropout) {
                activations = applyDropout(activations);
            }
            
            /* Set activations of neurons in the layer. */
            layer.setActivations(activations);
        }
    }
    
    /**
     * Applies the Rectified Linear Unit (ReLU) activation function to a weighted sum.
     * 
     * @param weightedSum	Weighted sum of a neuron.
     * @return	The activation value.
     */
    private double applyRelu(double weightedSum) {
        return Math.max(0, weightedSum);
    }
    
    /**
     * Calculates the derivative of the ReLU activation function.
     * 
     * @param activation	Activation of a neuron.
     * @return	The derivative of the activation value.
     */
    private double getReluDerivative(double activation) {
        return activation > 0 ? 1 : 0;
    }
    
    /**
     * This method uses softmax to calculate the activations of neurons in the output layer.
     * The activations represent a probability distribution where their sum is equal to 1.
     * 
     * @param weightedSums	Array of weighted sums.
     * @return	Array of activations representing a probability distribution.
     */
    private double[] applySoftmax(double[] weightedSums) {
        double exponentialSum = 0.0;
        double[] softmaxOutput = new double[weightedSums.length];
        
        /* Calculate exponential of each weighted sum and add it to the exponential sum. */
        for (int neuronIndex = 0; neuronIndex < weightedSums.length; neuronIndex++) {
            softmaxOutput[neuronIndex] = Math.exp(weightedSums[neuronIndex]);
            exponentialSum += softmaxOutput[neuronIndex];
        }
        
        /* Convert exponentials to probabilities. */
        for (int neuronIndex = 0; neuronIndex < softmaxOutput.length; neuronIndex++) {
            softmaxOutput[neuronIndex] /= exponentialSum;
        }
        return softmaxOutput;
    }
    
    /**
     * This method randomly sets the activation of a neuron to zero.
     * Used for regularization.
     * 
     * @param activations	Array of activations of neurons in a layer.
     * @return	Array of activations with certain neurons dropped.
     */
    private double[] applyDropout(double[] activations) {
        Random rand = new Random();
        
        /* Iterate through activations of neurons. */
        for (int neuronIndex = 0; neuronIndex < activations.length; neuronIndex++) {
        	
        	/* Randomly set the activation to zero when value generated is less than dropout rate. */
            if (rand.nextDouble() < this.dropoutRate) {
                activations[neuronIndex] = 0;
            }
        }
        return activations;
    }
    
    /**
     * This method is used to reset the error gradients for weights and biases in each layer
     * before back propagation.
     */
    private void resetGradients() {
    	
    	/* Iterate through each layer. */
        for (Layer layer : this.layers) {
        	
            /* Initialise empty arrays for resetting gradients. */
            double[] resetBiasGradients = new double[layer.getBiases().length];
            double[][] resetWeightGradients = new double[layer.getWeights().length][];
            
            /* Initialise empty sub-arrays for weights. */
            for (int index = 0; index < layer.getWeights().length; index++) {
                resetWeightGradients[index] = new double[layer.getWeights()[index].length];
            }
            
            /* Set empty gradient arrays in layer. */
            layer.setBiasGradients(resetBiasGradients);
            layer.setWeightGradients(resetWeightGradients);
        }
    }
    
    /**
     * Updates weights and biases based on error gradients calculated during back propagation. 
     * Stochastic Gradient Descent (SGD) optimiser is used to adjust the parameters in the direction
     * that minimises loss.
     */
    private void updateParameters() {
    	
    	/* Iterate through each layer. */
        for (Layer layer : this.layers) {
        	
        	/* Update bias and weights for each neuron based on the gradients and learning rate. */
            for (int neuron = 0; neuron < layer.getActivations().length; neuron++) {
            	
            	/* Calculate gradient step and subtract from bias for gradient descent. */
                layer.getBiases()[neuron] -= this.learningRate * layer.getBiasGradients()[neuron];
                for (int connection = 0; connection < layer.getWeights()[neuron].length; connection++) {
                	
                	/* Calculate gradient step and subtract from weight for gradient descent. */
                    layer.getWeights()[neuron][connection] -= this.learningRate * layer.getWeightGradients()[neuron][connection];
                }
                
            }
        }
    }
    
    /**
     * This method calculates the total entropy loss for a single sample after feed forward.
     * The loss represents how well the prediction matches the actual label.
     * A lower cross entropy loss indicates a better match.
     * 
     * @param predictedValues	Array of output layer activations.
     * @param digitLabel		Actual label of sample digit.
     * @return	The cross entropy loss.
     */
    private double calculateCrossEntropyLoss(double[] predictedValues, int digitLabel) {
        int numberOfClasses = 10;	// Number of classes in the output layer.

        /* Encode actual label with one-hot encoding. */
        double[] actualOneHot = new double[numberOfClasses];
        actualOneHot[digitLabel] = 1.0;

        /* Calculate cross-entropy loss. */
        double loss = 0.0;
        for (int index = 0; index < numberOfClasses; index++) {
            loss -= actualOneHot[index] * Math.log(predictedValues[index] + 1e-15);
        }
        return loss;
    }
    
    /**
     * This method returns the digit predicted by the model.
     * Activations represent probabilities, therefore the activation 
     * with the highest value is the predicted digit.
     * 
     * @param outputValues	Array of output layer activations.
     * @return	The predicted digit as an integer.
     */
    private int getPrediction(double[] outputValues) {
        int predictedClass = 0;
        double maxOutput = outputValues[0];
        for (int neuronIndex = 1; neuronIndex < outputValues.length; neuronIndex++) {
            if (outputValues[neuronIndex] > maxOutput) {
            	predictedClass = neuronIndex;
                maxOutput = outputValues[neuronIndex];
            }
        }
        return predictedClass;
    }
}