package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.qcri.ml4all.abstraction.api.UpdateLocal;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

public class NMFUpdate extends UpdateLocal<Double, Tuple2<Tuple2<INDArray, INDArray>, Tuple2<int[], Double>>> {
    double lower_bound;
    int index = 1;
    public NMFUpdate(double lower_bound) {
        this.lower_bound = lower_bound;

    }


    @Override
    public Double process(Tuple2<Tuple2<INDArray, INDArray>, Tuple2<int[], Double>> input, ML4allContext context) {
        int[] pointer = input.getField1().field0;
        int i = pointer[0];
        int j = pointer[1];
        System.out.println("update index : " + index + "   ===>  " + i + "," + j);

        index++;
        boolean updateWH = false;

        Tuple2<INDArray, INDArray> matrixs = input.getField0();
        if(matrixs.getField0() != null || matrixs.getField1() != null){
            INDArray w = (INDArray)context.getByKey("w");
            INDArray h = (INDArray)context.getByKey("h");


            INDArray aW = matrixs.getField0();
            INDArray aH = matrixs.getField1();

            INDArray updateW = w.getRow(i).add(aW);
            INDArray updateH = h.getColumn(j).add(aH);
            updateW = Transforms.max(updateW, lower_bound, true);
            updateH = Transforms.max(updateH, lower_bound, true);

            w.putRow(i, updateW);
            h.putColumn(j, updateH);

            context.put("w", w);
            context.put("h", h);

            return input.getField1().field1;
        }
        else{
            return 0.0;
        }


    }

    @Override
    public ML4allContext assign(Double input, ML4allContext context) {
        return context;
    }

}
