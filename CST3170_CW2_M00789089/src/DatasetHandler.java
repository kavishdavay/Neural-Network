import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The dataset handler reads, pre processes and returns data from a dataset using file name.
 * 
 * The system has been designed to handle a dataset where a row consists of 65 data points,
 * the first 64 represent pixel values ranging from 0 to 16 and the last is the label of the digit.
 * 
 * Due to the small size of the datasets, data augmentation is carried out by adding noise 
 * to each row of the dataset being used for training, thus doubling the size of the dataset. 
 * Pixel values are normalised to improve stability.
 * 
 * @version 1.0 22/02/2024
 * @author M00789089
 * 
 */
public class DatasetHandler {
 
    /**
     * This method reads data from a file, performs pre processing and returns a list of object arrays.
     * Each object consists of an array of normalised pixel values and the label of the corresponding digit.
     * Noise injection is performed to augment the training dataset.
     * Uses a flag to perform selective data augmentation on the training dataset only.
     * 
     * @param filename		Name of file.
     * @param noiseLevel	Level of noise for noise injection
     * @param isTraining	Flag for selective data augmentation (set to true for training set).
     * @return	The processed list of object arrays containing the data.
     */
    public static List<Object[]> readDataset(String filename, double noiseLevel, boolean isTraining) {
    	
        List<Object[]> data = new ArrayList<>();	// Initialise empty list of object arrays.
        
        /* Create instance of buffered reader and attempt to read file. */
        try (BufferedReader dataReader = new BufferedReader(new FileReader(filename))) {
        	
            String row;
            
            /* Go through each row in the file. */
            while ((row = dataReader.readLine()) != null) {
            	
                String[] values = row.split(",");	// Split values in row using comma separator and store in array.
                double[] input = new double[64]; 	// Initialise input array with a size of 64 to store pixel values.
                
                /* Convert first 64 values to double and store in input array. */
                for (int i = 0; i < 64; i++) {
                    input[i] = Double.parseDouble(values[i]);
                }
                
                int label = Integer.parseInt(values[64]);	// Last value is the label.
                
                /* Call method to normalise inputs and store in array. */
                double[] normalisedInputs = normaliseInputs(input);
                
                /* Check if file is being used for training. */
                if (isTraining) {
                	
                	/* Perform noise injection when true. */
                    double[] augmentedInputs = addNoise(normalisedInputs, noiseLevel);
                    
                    /* Both non-augmented and augmented inputs with their label are added to the list. 
                     * Each is treated as a distinct object in the list. */
                    data.add(new Object[]{normalisedInputs, label});
                    data.add(new Object[]{augmentedInputs, label});
                } else {
                	/* For testing dataset add only non-augmented inputs. */
                    data.add(new Object[]{normalisedInputs, label});
                }
            }
            
        } catch (FileNotFoundException e) {
        	
        	/* Throws runtime exception if file is not found. */
            throw new RuntimeException("File not found: " + filename);
        } catch (IOException e) {
        	
        	/* Throws runtime exception if there is an error reading file. */
            throw new RuntimeException("Error reading file: " + filename);
        }

        return data;
    }
    
    /**
     * Method to normalise input data points to the range 0 to 1.
     * 
     * @param inputs	Array of data points (Pixel values).
     * @return	Array of normalised input values.
     */
    private static double[] normaliseInputs(double[] inputs) {
        double[] normalisedInputs = new double[inputs.length];	// Initialise array to store normalised inputs.
        
        /* Iterate through input values, divide by 16, and store value in array. */
        for (int pixelIndex = 0; pixelIndex < inputs.length; pixelIndex++) {
            normalisedInputs[pixelIndex] = inputs[pixelIndex] / 16.0;
        }
        return normalisedInputs;
    }

    /**
     * Method adds noise to input values of a row.
     * 
     * @param inputs	Array of data points (Pixel Values).
     * @param noiseLevel Level of noise to use for injection.
     * @return	Array of inputs with added noise.
     */
    private static double[] addNoise(double[] inputs, double noiseLevel) {
    	Random rand = new Random();							// Create instance of random.
        double[] noisyInputs = new double[inputs.length];	// Initialise empty array to store noisy inputs.
        
        /* Iterate through input values, generate noise and add to input value. */
        for (int pixelIndex = 0; pixelIndex < inputs.length; pixelIndex++) {
            double noise = (rand.nextDouble() * 2 - 1) * noiseLevel;		// Generates a random noise value between -noiseLevel and +noiseLevel.
            noisyInputs[pixelIndex] = Math.min(Math.max(inputs[pixelIndex] + noise, 0), 1);	// Keeping values in range 0 to 1.
        }
        return noisyInputs;
    }
}