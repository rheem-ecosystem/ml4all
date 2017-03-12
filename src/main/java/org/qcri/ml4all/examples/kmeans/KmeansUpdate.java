package org.qcri.ml4all.examples.kmeans;

import org.qcri.ml4all.abstraction.api.Update;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.List;


/**
 * Created by zoi on 22/1/15.
 */
public class KmeansUpdate extends Update<Tuple2<Integer, double[]>, Tuple2<Integer, Tuple2<Integer, double[]>>> {


    @Override
    public Tuple2<Integer, double[]> process(Tuple2<Integer, Tuple2<Integer, double[]>> input, ML4allContext context) {
        int count = input.field1.field0;
        double[] newCenter = input.field1.field1;
        for (int j = 0; j < newCenter.length; j++) {
            newCenter[j] /= count;
        }
        return new Tuple2<>(input.field0, newCenter);
    }

    @Override
    public ML4allContext assign(List<Tuple2<Integer, double[]>> input, ML4allContext context) {
        double[][] centers = (double[][]) context.getByKey("centers");
        for (int i = 0; i < input.size(); i++) {
            Tuple2<Integer, double[]> c = input.get(i);
            int centroidId = c.field0;
            centers[centroidId] = c.field1;
        }
        context.put("centers", centers);
        return context;
    }

}
