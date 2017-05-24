package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.qcri.ml4all.abstraction.plan.ML4allPlan;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.net.MalformedURLException;
import java.util.Arrays;

import static org.qcri.ml4all.abstraction.plan.Platforms.*;

/**
 * Execute SGD for logistic regression.
 */
public class RunNMF {

    // Default parameters.

    static String relativePath = "src/main/resources/input/USCensus1990-NOMAD.txt";
    static int datasetSize  = 882;
    static int features = 68;
    static int k = 6;

    static int max_iterations = 4;
    static Platforms platform = SPARK_JAVA;

    static double regulizer = 0.0;
    static double lower_bound = 0.0;
    static double alfa= 0.0;
    static double beta = 0.0;

    public static void main (String... args) throws MalformedURLException {

        String propertiesFile = new File("src/main/resources/rheem.properties").getAbsoluteFile().toURI().toURL().toString();

        setClassVariables(args);

        String file = new File(relativePath).getAbsoluteFile().toURI().toURL().toString();

        System.out.println("max #maxIterations:" + max_iterations);

        long start_time = System.currentTimeMillis();

        ML4allPlan plan = new ML4allPlan();
        plan.setDatasetsize(datasetSize);
        char delimiter = ',';
        plan.setTransformOp(new NMFTransform(delimiter));
        plan.setLocalStage(new NMFStageWithRandomValues(k, datasetSize, features));
        plan.setSampleOp(new NMFSample());
        plan.setComputeOp(new NMFCompute(regulizer, datasetSize, features, alfa, beta));
        plan.setUpdateLocalOp(new NMFUpdate(lower_bound));
        plan.setLoopOp(new NMFLoop(max_iterations));

        ML4allContext context = plan.execute(file, platform, propertiesFile);
        System.out.println("Training finished in " + (System.currentTimeMillis() - start_time));
        System.out.println(context);
        printResult(context, file);
    }

    private static void printResult(ML4allContext context, String f){
        INDArray w = (INDArray)context.getByKey("w");
        INDArray h = (INDArray)context.getByKey("h");

        INDArray finalOut = w.mmul(h);
        Nd4j.writeTxt(finalOut, new File(f).getName() , ",");
    }
    private static void setClassVariables(String... args){
        if (args.length > 0) {
            relativePath = args[0];
            datasetSize = Integer.parseInt(args[1]);
            features = Integer.parseInt(args[2]);
            max_iterations = Integer.parseInt(args[3]);
            regulizer = Double.parseDouble(args[4]);
            alfa = Double.parseDouble(args[5]);
            beta = Double.parseDouble(args[6]);
            String platformIn = args[7];
            switch (platformIn) {
                case "spark":
                    platform = SPARK;
                    break;
                case "java":
                    platform = JAVA;
                    break;
                case "any":
                    platform = SPARK_JAVA;
                    break;
                default:
                    System.err.format("Unknown platform: \"%s\"\n", platform);
                    System.exit(3);
            }
        }
        else {
            System.out.println("Usage: java <main class> [<dataset path> <dataset size> <#features> <max maxIterations> <accuracy> <sample size>]");
            System.out.println("Loading default values");
        }

    }

}


