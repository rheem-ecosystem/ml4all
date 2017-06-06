package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.plan.ML4allPlan;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;


import static org.qcri.ml4all.abstraction.plan.Platforms.*;

/**
 * Execute SGD for nmf.
 */
public class RunNMF {
///Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandomTest.txt
    static String relativePath = "src/main/resources/input/apgRandomTest.txt";
    static int datasetSize  = 2509;
    static int features = 10499;
    static int k = 5;

    static int max_iterations = 5000;
    static Platforms platform = SPARK_JAVA;

    static double regulizer = 0.001;
    static double lower_bound = 0.0;
    static double alfa= 1;
    static double beta = 1;
    static double minRMSC = 0.001;

    public static void main (String... args) throws MalformedURLException {

       // new NMFTestMatrixFactorization();
        try {
            ml4AllConfig(args);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void printResult(ML4allContext context, String f){
        INDArray w = (INDArray)context.getByKey("w");
        INDArray h = (INDArray)context.getByKey("h");
        System.out.println("W final : " + w);
        System.out.println("H final : " + h);
        INDArray finalOut = w.mmul(h);
        System.out.println("finalOut : " + finalOut);
        Nd4j.writeTxt(finalOut, new File(f).getName() );


    }

    private static void ml4AllConfig(String... args) throws Exception{
         String propertiesFile = new File("src/main/resources/rheem.properties").getAbsoluteFile().toURI().toURL().toString();

         try {

             String path = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandomTest.txt";

             setClassVariables(args);

             String file = new File(relativePath).getAbsoluteFile().toURI().toURL().toString();

             System.out.println("max #maxIterations:" + max_iterations);

             long start_time = System.currentTimeMillis();

             ML4allPlan plan = new ML4allPlan();
             plan.setDatasetsize(datasetSize);
             char delimiter = ',';
             plan.setTransformOp(new NMFTransform(delimiter));
             plan.setLocalStage(new NMFStageWithRandomValues(k, path, delimiter));
             plan.setSampleOp(new NMFSample());
             plan.setComputeOp(new NMFCompute(regulizer, alfa, beta));
             plan.setUpdateLocalOp(new NMFUpdate(lower_bound));
             plan.setLoopOp(new NMFLoop(max_iterations, minRMSC));

             ML4allContext context = plan.execute(file, platform, propertiesFile);
             System.out.println("Training finished in " + (System.currentTimeMillis() - start_time));
             System.out.println(context);
             printResult(context, file);

         } catch (IOException e) {
            e.printStackTrace();
         }

    }

    private static int calculateOmegaTest(INDArray testDocument) {
        int nonZeroValueCount = 0;
        for (int i = 0; i < testDocument.rows(); i++) {
            for (int j = 0; j < testDocument.columns(); j++) {
                if (testDocument.getDouble(i, j) > 0.0) {
                    nonZeroValueCount++;
                }
            }
        }

        return nonZeroValueCount;
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


