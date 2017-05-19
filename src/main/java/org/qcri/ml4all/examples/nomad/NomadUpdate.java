package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.UpdateLocal;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

public class NomadUpdate extends UpdateLocal<Tuple2<INDArray, INDArray>, Tuple2<INDArray, INDArray>> {

    public NomadUpdate() { }


    @Override
    public Tuple2<INDArray, INDArray> process(Tuple2<INDArray, INDArray> input, ML4allContext context) {
        INDArray h = (INDArray) context.getByKey("h");
        INDArray w = (INDArray) context.getByKey("w");
        int i = (int)context.getByKey("i");
        int j = (int)context.getByKey("j");

        w.putRow(i, input.getField0());
        h.putColumn(j, input.getField1());

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
