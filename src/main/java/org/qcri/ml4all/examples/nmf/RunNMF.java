package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.plan.ML4allPlan;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.sql.Timestamp;
import java.util.Date;


import static org.qcri.ml4all.abstraction.plan.Platforms.*;

/**
 * Execute SGD for nmf.
 */
public class RunNMF {
///Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgNoHeaderInfoBigTable.txt
    ///Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_big_table.csv
    static String relativePath = "src/main/resources/input/apg_big_table.csv";
    static String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/";
    static String path = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_big_table.csv";
    static int datasetSize  = 2509;
    static int features = 10499;
    static int k = 5;

    static Platforms platform = JAVA;

    static double lower_bound = 0.0;
    static double beta = 0.01;
    static double minRMSC = 0.001;
    static int seed = 1234;
    static int epoch = 10;

    public static void main (String... args) throws MalformedURLException {

        try {
            ml4AllConfig(args);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void printResult(ML4allContext context, String f){
        try{
            System.out.println(context);

            INDArray w = (INDArray)context.getByKey("w");
            INDArray h = (INDArray)context.getByKey("h");
            INDArray R = w.mmul(h);

            System.out.println(R.getRow(0));

            File fs4 = new File(outputPath +"apg_nmf_ml4all_all.bin");
            File fs4w = new File(outputPath +"apg_nmf_ml4all_all_w.bin");
            File fs4h = new File(outputPath +"apg_nmf_ml4all_all_h.bin");

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
             setClassVariables(args);

             String file = new File(relativePath).getAbsoluteFile().toURI().toURL().toString();

             long start_time = System.currentTimeMillis();

             ML4allPlan plan = new ML4allPlan();
             plan.setDatasetsize(datasetSize);
             char delimiter = ',';
             plan.setTransformOp(new NMFTransform(delimiter));
             plan.setLocalStage(new NMFStageWithRandomValues(k,seed, datasetSize, features));
             plan.setSampleOp(new NMFSample(1));
             plan.setComputeOp(new NMFCompute(beta, k));
             plan.setUpdateLocalOp(new NMFUpdate(lower_bound));
             plan.setLoopOp(new NMFLoop(datasetSize, features, minRMSC, epoch));


             System.out.println(new Timestamp(new Date().getTime()));

             ML4allContext context = plan.execute(file, platform, propertiesFile);
             System.out.println("Training finished in " + (System.currentTimeMillis() - start_time));
             System.out.println(new Timestamp(new Date().getTime()));
             printResult(context, file);

         } catch (IOException e) {
            e.printStackTrace();
         }

    }

    private static void setClassVariables(String... args){
        if (args.length > 0) {
            relativePath = args[0];
            String platformIn = args[1];
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
            INDArray documentMaster = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_big_table.csv", ",");
            datasetSize = documentMaster.rows();
            features = documentMaster.columns();
        } catch (IOException e) {
            e.printStackTrace();
        }



    }

}


