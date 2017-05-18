package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

public class NomadCompute extends Compute<Tuple2, Tuple2<Tuple2<Integer, Integer>, Double>> {

    double stepSize = 1.0;
    double regulizer = 1.0;

    public NomadCompute(double stepSize, double regulizer) {
        this.stepSize = stepSize;
        this.regulizer = regulizer;
    }

    @Override
    public Tuple2 process(Tuple2<Tuple2<Integer, Integer>, Double> input, ML4allContext context) {

        INDArray w = (INDArray) context.getByKey("w");
        INDArray h = (INDArray) context.getByKey("h");

        double aDataPoint = input.getField1();
        int i = input.getField0().field0;
        int j = input.getField0().field1;


        double aW = aDataPoint - w.getRow(i).mmul(h.getColumn(j)).getDouble(0);
        INDArray updateW =h.getRow(j).muli(aW);
        updateW = updateW.add(w.getColumn(i).muli(regulizer));
        updateW = updateW.muli(stepSize);
        updateW = w.getColumn(i).subi(updateW);

        double aH = aDataPoint - w.getRow(i).mmul(h.getColumn(j)).getDouble(0) ;
        INDArray updateH =w.getColumn(j).muli(aW);
        updateH = updateH.add(h.getRow(j).muli(regulizer));
        updateH = updateH.muli(stepSize);
        updateH = h.getRow(j).subi(updateH);


        return new Tuple2(updateW, updateH);
    }
}
