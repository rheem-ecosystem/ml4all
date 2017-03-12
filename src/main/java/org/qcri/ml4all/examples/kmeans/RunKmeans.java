package org.qcri.ml4all.examples.kmeans;

import org.qcri.ml4all.abstraction.plan.ML4allPlan;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.net.MalformedURLException;
import java.util.Arrays;

import static org.qcri.ml4all.abstraction.plan.Platforms.SPARK_JAVA;

/**
 * Created by zoi on 22/1/15.
 */
public class RunKmeans {

    public static void main(String[] args) throws MalformedURLException {

        int numberOfCentroids = 3;
//        String file = "fsrc/main/resources/input/kmeans_data.txt";
//        int dimension = 3;
        String relativePath = "src/main/resources/input/USCensus1990-sample.txt";
        String file = new File(relativePath).getAbsoluteFile().toURI().toURL().toString();
        String propertiesFile = new File("rheem.properties").getAbsoluteFile().toURI().toURL().toString();
        int dimension = 68;
        double accuracy = 0;
        int maxIterations = 10;
        Platforms platform = SPARK_JAVA;

        //Usage:  <data_file> <#points> <#dimension> <#centroids> <accuracy>
        if (args.length > 0) {
            file = args[0];
            dimension = Integer.parseInt(args[1]);
            numberOfCentroids = Integer.parseInt(args[2]);
            accuracy = Double.parseDouble(args[3]);
            maxIterations = Integer.parseInt(args[4]);
            String platformIn = args[5];
        }
        else {
            System.out.println("Loading default values");
        }

        long start_time = System.currentTimeMillis();

        ML4allPlan plan = new ML4allPlan();

        //logical operators of template
        plan.setTransformOp(new TransformCSV());
        plan.setLocalStage(new KmeansStageWithZeros(numberOfCentroids, dimension));
        plan.setComputeOp(new KmeansCompute());
        plan.setUpdateOp(new KmeansUpdate());
        plan.setLoopOp(new KmeansConvergeOrMaxIterationsLoop(accuracy, maxIterations));

        ML4allContext context = plan.execute(file, platform, propertiesFile);
        System.out.println("Centers:" + Arrays.deepToString((double [][])context.getByKey("centers")));

        System.out.println("Total time: " + (System.currentTimeMillis() - start_time));
    }
}
