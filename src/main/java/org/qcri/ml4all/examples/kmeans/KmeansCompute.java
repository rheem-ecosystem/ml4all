package org.qcri.ml4all.examples.kmeans;


import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.rheem.basic.data.Tuple2;

/**
 * Created by zoi on 22/1/15.
 */
public class KmeansCompute extends Compute<Tuple2<Integer, Tuple2<Integer, double[]>>, double[]> {

    @Override
    public Tuple2 process(double[] input, ML4allContext context) {
        double[][] centers = (double[][]) context.getByKey("centers");
        double min = Double.MAX_VALUE;
        int minIndex = 0;
        for (int i = 0; i < centers.length; i++) {
            double dist = dist(input, centers[i]);
            if (dist < min) {
                min = dist;
                minIndex = i;
            }
        }

        return new Tuple2(minIndex, new Tuple2<>(1, input)); //groupby clusterID, and count
    }

    @Override
    public Tuple2<Integer, Tuple2<Integer, double[]>> aggregate(Tuple2<Integer, Tuple2<Integer, double[]>> input1, Tuple2<Integer, Tuple2<Integer, double[]>> input2) {
        Tuple2<Integer, double[]> kv1 = input1.field1;
        Tuple2<Integer, double[]> kv2 = input2.field1;
        int count = kv1.field0 + kv2.field0;
        double[] sum = new double[kv1.field1.length];
        for (int i = 0; i < kv1.field1.length; i++)
            sum[i] = kv1.field1[i] + kv2.field1[i];
        return new Tuple2(input1.field0, new Tuple2<>(count, sum));
    }

    private double dist(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++)
            sum += (a[i] - b[i])*(a[i] - b[i]);
        return Math.sqrt(sum);
    }
}
