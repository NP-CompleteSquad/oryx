package com.cloudera.oryx.kmeans.common;

import com.cloudera.oryx.common.math.Vectors;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

/**
 * Reads a file with numeric values and forms d-dimensional Weighted Vectors (initially with equal weights=1).
 * Used for quick unit testing in large scale with random data-points.
 */
public class VectorInputData {

    /**
     * Gathers up the numeric values into a list of Doubles from the specified file.
     * @param file the given file
     * @return
     * @throws FileNotFoundException
     */
    private List<Double> getVectorCoordinates(File file) throws FileNotFoundException {
        Scanner vectorDataFile = new Scanner(file);
        List<Double> vectorCoordinates = Lists.newArrayList();
        while (vectorDataFile.hasNextLine()) {
            String line = vectorDataFile.nextLine();
            Scanner scanner = new Scanner(line);
            scanner.useLocale(Locale.US);
            scanner.useDelimiter("\\s+|,");
            while (scanner.hasNextDouble()) {
                vectorCoordinates.add(scanner.nextDouble());
            }
            scanner.close();
        }
        vectorDataFile.close();
        return vectorCoordinates;
    }

    /**
     * Partitioning the {@code List<Double>} vector coordinates into d-dimensional parts.
     * @param dimensions the dimensions of the vectors
     * @param file path of the file with the numeric values to be turned into vectors
     * @return equals d-dimensional weighted vectors of weight=1
     * @throws FileNotFoundException
     */
    public List<WeightedRealVector> getTestVectors(int dimensions, File file) throws FileNotFoundException {
        Preconditions.checkArgument(dimensions>0,"dimension number must be an integer greater than zero");
        Random rand = new Random();
        List<WeightedRealVector> vectors = Lists.newArrayList();
        List<Double> vecCoordinates = getVectorCoordinates(file);
        int mod = (vecCoordinates.size() % dimensions);
        int numVectors = (int) (vecCoordinates.size() / dimensions);
        for (int i = 0; i < numVectors; i++) {
            double[] vec = new double[dimensions];
            for (int j = 0; j < dimensions; j++) {
                vec[j] = vecCoordinates.get((i * dimensions) + j);
            }
            vectors.add(new WeightedRealVector(Vectors.of(vec), 1));
        }
        if (mod != 0) {
            double[] vec = new double[dimensions];
            //Take the values left in the file that do not form a vec of d-dimensions.
            for (int j = 0; j < (dimensions - mod); j++) {
                vec[j] = vecCoordinates.get((numVectors * dimensions) + j);
            }
            //Fill randomly with previous coordinates the rest of the missing values-coordinates to form a d-dimensional vec.
            for (int l = 0; l < (dimensions - (dimensions - mod)); l++) {
                vec[(dimensions - mod) + l] = vecCoordinates.get(rand.nextInt(numVectors*dimensions)+1);
            }
            vectors.add(new WeightedRealVector(Vectors.of(vec), 1));
        }
        return vectors;
    }
}