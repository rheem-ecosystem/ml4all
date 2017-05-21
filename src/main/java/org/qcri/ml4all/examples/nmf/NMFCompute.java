package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.Random;

public class NMFCompute extends Compute<Tuple2, INDArray> {

    double stepSize = 1.0;
    double regulizer = 1.0;
    int m;
    int n;
    Random rand;
    public NMFCompute(double stepSize, double regulizer, int m, int n) {
        this.stepSize = stepSize;
        this.regulizer = regulizer;
        this.m = m;
        this.n = n;
        this.rand = new Random();
    }

    @Override
    public Tuple2 process(INDArray input , ML4allContext context) {

        INDArray w = (INDArray) context.getByKey("w");
        INDArray h = (INDArray) context.getByKey("h");


        int i = rand.nextInt(this.m);
        int j = rand.nextInt(this.n);

        double aDataPoint = input.getDouble(i,j);

        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));
        INDArray updateW =h.getColumn(j).mul(aW).transpose();
        updateW = updateW.add(w.getRow(i).mul(regulizer));
        updateW = updateW.mul(stepSize);

        double aH = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0)) ;
        INDArray updateH =w.getRow(i).mul(aH).transpose();
        updateH = updateH.add(h.getColumn(j).mul(regulizer));
        updateH = updateH.mul(stepSize);

        context.put("i",i);
        context.put("j",j);

        return new Tuple2(updateW, updateH);
    }
}
