import java.util.List;

/**
 * The main class for the digit recognition task.
 * An Artificial Neural Network with 2 hidden layers is trained and evaluated on the datasets.
 * 
 * @version 1.0 22/02/2024
 * @author M00789089
 * 
 */
public class Main {

	/* File names */
	private static final String firstDatasetName = "cw2DataSet1.csv";
	private static final String secondDatasetName = "cw2DataSet2.csv";
	
	/* Hyper-parameters */
	private static final int INPUT_LAYER_SIZE = 64;				// Number of input features in the input layer.
	private static final int[] LAYER_SIZES = {256, 256, 10};	// Number of neurons in first hidden layer, second hidden layer and output layer.
	private static final int EPOCHS = 90;						// Number of epochs to train model.
	private static final double LEARNING_RATE = 0.006;			// Learning rate for gradient descent.
	private static final double DROPOUT_RATE = 0.4;				// Rate for neuron drop out.
	private static final double NOISE_LEVEL = 0.3;				// Level of noise for noise injection.

	
    /**
     * This is the main method of the system. 
     * It carries out the two fold test by initialising two models.
     * The datasets used for training/evaluation are swapped for each model.
     * Stores the test and train accuracies for each model.
     * 
     * @param args
     */
    public static void main(String[] args) {
        
        /* Perform first test. */
        System.out.println("Training First Model\n");
        double[] firstTestResults = runModel(firstDatasetName, secondDatasetName);
        
        /* Perform second test. */
        System.out.println("\nTraining Second Model\n");
        double[] secondTestResults = runModel(secondDatasetName, firstDatasetName);
        
        /* Print results of the two fold test. */
        printResults(firstTestResults, secondTestResults);
    }
    
    /**
     * This method returns the train and test accuracy after building, training and evaluating the model.
     * Uses the dataset handler to read and process the files containing the training and testing sets.
     * 
     * @param trainFilename Name of dataset for training.
     * @param testFilename 	Name of dataset for testing.
     * @return A double array containing train accuracy at index 0 and test accuracy at index 1.
     */
    private static double[] runModel(String trainFilename, String testFilename) {
        
    	/* Reading and storing datasets used for training and testing. */
        List<Object[]> trainingDataset = DatasetHandler.readDataset(trainFilename, NOISE_LEVEL, true);
        List<Object[]> testingDataset = DatasetHandler.readDataset(testFilename, NOISE_LEVEL, false);
        
        /* Building and running the neural network. */
        NeuralNetwork neuralNetwork = new NeuralNetwork(INPUT_LAYER_SIZE, LAYER_SIZES, EPOCHS, LEARNING_RATE, DROPOUT_RATE);
        neuralNetwork.train(trainingDataset);        						// Training model with training set.
        double trainAccuracy = neuralNetwork.evaluate(trainingDataset);    	// Evaluating model on training set.
        double testAccuracy = neuralNetwork.evaluate(testingDataset);       // Evaluating model on unseen testing set.

        double[] accuracyResults = {trainAccuracy, testAccuracy};
        
        return accuracyResults;
    }
    
    /**
     * Method to print out results for the two fold test.
     * Calculates the average test and train accuracies.
     * 
     * @param firstTestResults		Array of train and test accuracies for the first model.
     * @param secondTestResults		Array of train and test accuracies for the second model.
     */
    private static void printResults(double[] firstTestResults, double[] secondTestResults) {
    	System.out.println("\n--------------------------------------------------");
    	System.out.println("                     Results");
    	System.out.println("--------------------------------------------------\n");
    	
    	/* Print first model results. */
        System.out.println("First Model");
        System.out.println("Training Set: " + firstDatasetName);
        System.out.println("Testing Set: " + secondDatasetName);
        System.out.println("Train Accuracy: " + String.format("%.3f", firstTestResults[0]) + "%");
        System.out.println("Test Accuracy: " + String.format("%.3f", firstTestResults[1]) + "%\n");
        
        /* Print Second Model Results. */
        System.out.println("Second Model");
        System.out.println("Training Set: " + secondDatasetName);
        System.out.println("Testing Set: " + firstDatasetName);
        System.out.println("Train Accuracy: " + String.format("%.3f", secondTestResults[0]) + "%");
        System.out.println("Test Accuracy: " + String.format("%.3f", secondTestResults[1]) + "%\n");
        
        /* Calculate average train and test accuracies. */
        double averageTestAccuracy = (firstTestResults[1] + secondTestResults[1]) / 2;
        double averageTrainAccuracy = (firstTestResults[0] + secondTestResults[0]) / 2;
        
        /* Print system performance. */
        System.out.println("\nSystem Performance\n");
        System.out.println("Average Train Accuracy: " + String.format("%.3f", averageTrainAccuracy) + "%");
        System.out.println("Average Test Accuracy: " + String.format("%.3f", averageTestAccuracy) + "%");
    }
}