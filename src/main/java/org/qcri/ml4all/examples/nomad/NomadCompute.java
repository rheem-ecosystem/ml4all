package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.Random;

public class NomadCompute extends Compute<Tuple2, INDArray> {

    double stepSize = 1.0;
    double regulizer = 1.0;

    public NomadCompute(double stepSize, double regulizer) {
        this.stepSize = stepSize;
        this.regulizer = regulizer;
    }

    @Override
    public Tuple2 process(INDArray input , ML4allContext context) {

        INDArray w = (INDArray) context.getByKey("w");
        INDArray h = (INDArray) context.getByKey("h");

        Random rand = new Random();
        int i = rand.nextInt((int) context.getByKey("m"));
        int j = rand.nextInt((int) context.getByKey("n"));
        double aDataPoint = input.getDouble(i,j);

        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));
        INDArray updateW =h.getColumn(j).mul(aW).transpose();
        updateW = updateW.add(w.getRow(i).mul(regulizer));
        updateW = updateW.mul(stepSize);
        updateW = w.getRow(i).sub(updateW);

        double aH = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0)) ;
        INDArray updateH =w.getRow(i).mul(aW).transpose();
        updateH = updateH.add(h.getColumn(j).mul(regulizer));
        updateH = updateH.mul(stepSize);
        updateH = h.getColumn(j).sub(updateH);

        context.put("i",i);
        context.put("j",j);

        return new Tuple2(updateW, updateH);
    }
}
