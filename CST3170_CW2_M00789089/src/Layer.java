import java.util.Random;

/**
 * This class represents a layer in the neural network.
 * Used to store weighted sums, activations, weights, biases, and the error gradients.
 * Has getter/setter methods and a method to calculate weighted sums of neurons in the layer.
 * 
 * @version 1.0 22/02/2024
 * @author M00789089
 * 
 */
public class Layer {
	private int numberOfNeurons;			// Number of neurons in the layer.
	private double[] weightedSums;			// Array of weighted sums.
	private double[] activations;			// Array of activation values.
    private double[][] weights;				// 2D array for synaptic weights between neurons of current and previous layer.
    private double[] biases;				// Array for biases of neurons in the layer.
    private double[][] weightGradients;		// 2D array for weight gradients calculated during back propagation.
    private double[] biasGradients; 		// Array for bias gradients calculated during back propagation.
    
    /**
     * Constructor method to create a layer instance.
     * Randomly initialises weights/biases using the Kaiming initialisation method.
     * 
     * @param numberOfNeurons	Number of neurons in the layer.
     * @param numberOfInputs	Number of input neurons to the layer.
     */
    public Layer(int numberOfNeurons, int numberOfInputs) {
    	this.numberOfNeurons = numberOfNeurons;
    	this.weightedSums = new double[numberOfNeurons];
    	this.activations = new double[numberOfNeurons];
        this.weights = new double[numberOfNeurons][numberOfInputs];
        this.biases = new double[numberOfNeurons];
        this.weightGradients = new double[numberOfNeurons][numberOfInputs];
        this.biasGradients = new double[numberOfNeurons];

        Random rand = new Random();
        
        /* Calculate the standard deviation for weight initialisation based on Kaiming method. */
        double standardDeviation = Math.sqrt(2.0 / numberOfInputs);
        
        /* Iterate over each neuron in layer to initialise weights/biases. */
        for (int neuronIndex = 0; neuronIndex < numberOfNeurons; neuronIndex++) {
        	
            biases[neuronIndex] = 0;	// Initialise biases to zero.
            
            /* Initialise weights by applying standard deviation to random values generated using Gaussian Distribution. */
            for (int connectionIndex = 0; connectionIndex < numberOfInputs; connectionIndex++) {
                weights[neuronIndex][connectionIndex] = rand.nextGaussian() * standardDeviation;
            }
        }
    }
    
    /* Getter Methods. */
    public double[] getWeightedSums() {
    	return this.weightedSums;
    }
    
    public double[] getActivations() {
    	return this.activations;
    }
    
    public double[][] getWeights() {
    	return this.weights;
    }
    
    public double[] getBiases() {
    	return this.biases;
    }
    
    public double[][] getWeightGradients() {
    	return this.weightGradients;
    }
    
    public double[] getBiasGradients() {
    	return this.biasGradients;
    }

    /* Setter Methods. */
    public void setWeightedSums(double[] weightedSums) {
    	this.weightedSums = weightedSums;
    }
    
    public void setActivations(double[] activationValues) {
    	this.activations = activationValues;
    }
    
    public void setWeightGradients(double[][] calculatedWeightGradients) {
    	this.weightGradients = calculatedWeightGradients;
    }
    
    public void setBiasGradients(double[] calculatedBiasGradients) {
    	this.biasGradients = calculatedBiasGradients;
    }
    
    /**
     * This method calculates the weighted sums of neurons in the layer using the input values, weights and biases.
     * 
     * @param inputValues Values of neurons from previous layer.
     */
    public void calculateWeightedSums(double[] inputValues) {
    	
    	/* Iterate through each neuron in the layer and calculate its weighted sum. */
        for (int currentNeuron = 0; currentNeuron < this.numberOfNeurons; currentNeuron++) {
            double sum = 0.0;																		// Initialise sum.
            
            /* Iterate over each input value, apply weight, and add to sum. */
            for (int previousNeuron = 0; previousNeuron < inputValues.length; previousNeuron++) {
                sum += this.weights[currentNeuron][previousNeuron] * inputValues[previousNeuron];    
            }
            sum += this.biases[currentNeuron];														// Add bias to sum.
            this.weightedSums[currentNeuron] = sum;													// Store sum for neuron i.
        }
    }
    
}