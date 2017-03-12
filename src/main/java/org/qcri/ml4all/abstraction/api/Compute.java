package org.qcri.ml4all.abstraction.api;

import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class Compute<R, V> extends LogicalOperator {

    /**
     * Performs a computation at the data unit granularity
     *
     * @param input a data unit
     * @param context
     */
    public abstract R process(V input, ML4allContext context);

    /**
     * Aggregates the output of the process() method to use in a group by
     */
//    public abstract R aggregate(R input1, R input2);

    //default implementation used in GD algorithms
    public R aggregate(R input1, R input2) { //for now, we assume the gradient is always a double[]
        double[] g1 = (double[]) input1;
        double[] g2 = (double[]) input2;

        if (g2 == null) //samples came from one partition only
            return (R) g1;

        if (g1 == null) //samples came from one partition only
            return (R) g2;

        double[] sum = new double[g1.length];
        sum[0] = g1[0] + g2[0]; //count
        for (int i = 1; i < g1.length; i++)
            sum[i] = g1[i] + g2[i];

//        System.out.println("sum:" + Arrays.toString(sum));
        return (R) sum;
    }

}
