package org.qcri.ml4all.examples.sgd;

import org.qcri.ml4all.abstraction.plan.ML4allPlan;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.net.MalformedURLException;
import java.util.Arrays;

import static org.qcri.ml4all.abstraction.plan.Platforms.*;

/**
 * Execute SGD for logistic regression.
 */
public class RunSGD {

    // Default parameters.
    static String relativePath = "src/main/resources/input/adult.zeros";
    static int datasetSize  = 100827;
    static int features = 123;

    //these are for SGD/mini run to convergence
    static double accuracy = 0.1;
    static int max_iterations = 10;
    static Platforms platform = SPARK_JAVA;


    public static void main (String... args) throws MalformedURLException {

        String propertiesFile = new File("rheem.properties").getAbsoluteFile().toURI().toURL().toString();

        //Usage: <data_file> <#features> <sparse> <binary>
        if (args.length > 0) {
            relativePath = args[0];
            datasetSize = Integer.parseInt(args[1]);
            features = Integer.parseInt(args[2]);
            max_iterations = Integer.parseInt(args[3]);
            accuracy = Double.parseDouble(args[4]);
            String platformIn = args[5];
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

        String file = new File(relativePath).getAbsoluteFile().toURI().toURL().toString();

        System.out.println("max #maxIterations:" + max_iterations);
        System.out.println("accuracy:" + accuracy);

        long start_time = System.currentTimeMillis();
        ML4allPlan plan = new ML4allPlan();
        plan.setDatasetsize(datasetSize);

        //logical operators of template
        plan.setTransformOp(new LibSVMTransform(features));
        plan.setLocalStage(new SGDStageWithZeros(features));
//        plan.setSampleOp(new SGDSample());
        plan.setComputeOp(new ComputeLogisticGradient());
        plan.setUpdateLocalOp(new WeightsUpdate());
        plan.setLoopOp(new SGDLoop(accuracy, max_iterations));

        ML4allContext context = plan.execute(file, platform, propertiesFile);
        System.out.println("Training finished in " + (System.currentTimeMillis() - start_time));
        System.out.println(context);
        System.out.println("Weights:" + Arrays.toString((double [])context.getByKey("weights")));
    }

}


