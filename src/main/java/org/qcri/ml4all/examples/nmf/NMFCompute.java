package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.HashMap;
import java.util.Map;


public class NMFCompute extends Compute<Tuple2, double[]> {

    private double regulizer;
    private double alpha;
    private double beta;

    private static Map<String, Integer> indexIter;

    public NMFCompute(double regulizer,double alpha, double beta) {
        this.regulizer = regulizer;
        this.alpha = alpha;
        this.beta = beta;

        indexIter = new HashMap<>();

    }

    @Override
    public Tuple2 process(double[] input , ML4allContext context) {
        INDArray updateW = null;
        INDArray updateH = null;

        INDArray w = (INDArray) context.getByKey("w");
        INDArray h = (INDArray) context.getByKey("h");

        int i = (int)input[0];
        int j = (int)input[1];

        double aDataPoint = input[2];
        double stepSize = this.getStepSize(context, i, j);

        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));

        updateW = w.getRow(i).sub(
                (
                        ((h.getColumn(j).mul(aW))
                                .add(
                                        (
                                                w.getRow(i).mul(this.regulizer)
                                        ).transpose()
                                ))
                                .mul(stepSize)
                )
        );

        double aWH = aDataPoint - (updateW.mmul(h.getColumn(j)).getDouble(0));

        updateH =((w.getRow(i).mul(aWH))
                    .add(
                            (
                                    h.getColumn(j).mul(this.regulizer)
                            ).transpose()
                    )
                )
                .mul(stepSize);

        context.put("i",i);
        context.put("j",j);

        return new Tuple2(updateW, updateH);
    }


    private double getStepSize(ML4allContext context, int i, int j){

        String key = i+","+j;
        int updatedIteration = 0;

        if(indexIter.containsKey(key)){
            updatedIteration = indexIter.get(key);
        }
        updatedIteration++;

        indexIter.put(key, updatedIteration);

        return this.alpha / (1+ (this.beta * Math.pow(updatedIteration, 1.5))) ;
    }


}
