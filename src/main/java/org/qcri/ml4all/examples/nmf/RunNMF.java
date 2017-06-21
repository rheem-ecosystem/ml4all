package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.plan.ML4allPlan;
import org.qcri.ml4all.abstraction.plan.ML4allPlanNew;
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
    static String relativePath = "src/main/resources/input/apgNoHeaderInfoBigTable.txt";
    static int datasetSize  = 2509;
    static int features = 10499;
    static int k = 5;

    static int max_iterations = datasetSize * features;
    static Platforms platform = JAVA;

    static double lower_bound = 0.0;
    static double alfa= 1;
    static double beta = 0.01;
    static double minRMSC = 0.001;
    static int seed = 1234;
    static int epoch = 400;
    public static void main (String... args) throws MalformedURLException {

        try {
            ml4AllConfig(args);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void printResult(ML4allContext context, String f){
        try{
            String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/";

            INDArray w = (INDArray)context.getByKey("w");
            INDArray h = (INDArray)context.getByKey("h");
            INDArray R = w.mmul(h);

            System.out.println(R.getRow(0));

            File fs4 = new File(outputPath +"apg_nmf_all.bin");
            File fs4w = new File(outputPath +"apg_nmf_all_w.bin");
            File fs4h = new File(outputPath +"apg_nmf_all_h.bin");

            Nd4j.writeTxt(R, new File(f).getName() );

            Nd4j.saveBinary(R, fs4);
            Nd4j.saveBinary(w, fs4w);
            Nd4j.saveBinary(h, fs4h);
        }
        catch(Exception e){
            System.out.println(e);
        }
    }


    private static void ml4AllConfig(String... args) throws Exception{
         String propertiesFile = new File("src/main/resources/rheem.properties").getAbsoluteFile().toURI().toURL().toString();

         try {

             //String path = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandomTest.txt";
             String path = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgNoHeaderInfoBigTable.txt";

             setClassVariables(args);

             String file = new File(relativePath).getAbsoluteFile().toURI().toURL().toString();

             System.out.println("max #maxIterations - epoch:" + max_iterations);

             long start_time = System.currentTimeMillis();

             ML4allPlan plan = new ML4allPlan();
             plan.setDatasetsize(datasetSize);
             char delimiter = ',';
             plan.setTransformOp(new NMFTransform(delimiter));
             plan.setLocalStage(new NMFStageWithRandomValues(k, path, delimiter, seed));
             //plan.setLocalStage(new NMFStageWithRandomValues(k,seed, datasetSize, features));
             plan.setSampleOp(new NMFSample(1));
             plan.setComputeOp(new NMFCompute(beta, k));
             plan.setUpdateLocalOp(new NMFUpdate(lower_bound));
             //plan.setUpdateOp(new NMFUpdate(lower_bound));
             plan.setLoopOp(new NMFLoop(datasetSize, features, minRMSC, epoch));

             ML4allContext context = plan.execute(file, platform, propertiesFile);
             System.out.println("Training finished in " + (System.currentTimeMillis() - start_time));
             System.out.println(context);
             printResult(context, file);

         } catch (IOException e) {
            e.printStackTrace();
         }

    }

    private static void setClassVariables(String... args){
        if (args.length > 0) {
            relativePath = args[0];
            datasetSize = Integer.parseInt(args[1]);
            features = Integer.parseInt(args[2]);
            max_iterations = Integer.parseInt(args[3]);
            alfa = Double.parseDouble(args[4]);
            beta = Double.parseDouble(args[5]);
            String platformIn = args[6];
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

        try {
            INDArray documentMaster = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgNoHeaderInfoBigTable.txt", ",");
            datasetSize = documentMaster.rows();
            features = documentMaster.columns();
            max_iterations = datasetSize * features;
        } catch (IOException e) {
            e.printStackTrace();
        }



    }

}


