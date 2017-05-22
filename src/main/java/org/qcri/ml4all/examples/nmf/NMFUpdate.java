package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.qcri.ml4all.abstraction.api.UpdateLocal;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

public class NMFUpdate extends UpdateLocal<Tuple2<INDArray, INDArray>, Tuple2<INDArray, INDArray>> {
    double max;
    public NMFUpdate(double max) {
        this.max = max;
    }


    @Override
    public Tuple2<INDArray, INDArray> process(Tuple2<INDArray, INDArray> input, ML4allContext context) {
        INDArray h = (INDArray) context.getByKey("h");
        INDArray w = (INDArray) context.getByKey("w");
        int i = (int)context.getByKey("i");
        int j = (int)context.getByKey("j");

        INDArray updateW = w.getRow(i).sub(input.getField0());
        updateW = Transforms.max(updateW, max, true);

        INDArray updateH = h.getColumn(j).sub(input.getField1());
        updateH = Transforms.max(updateH, max, true);

        w.putRow(i, updateW);
        h.putColumn(j, updateH);

        return new Tuple2<>(w, h);
    }

    @Override
    public ML4allContext assign(Tuple2<INDArray, INDArray> input, ML4allContext context) {
        context.put("w", input.field0);
        context.put("h", input.field1);
        int iteration = (int) context.getByKey("iter");
        context.put("iter", ++iteration);
        return context;
    }



}
