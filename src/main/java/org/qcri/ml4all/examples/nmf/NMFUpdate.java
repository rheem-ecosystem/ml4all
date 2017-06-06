package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.qcri.ml4all.abstraction.api.UpdateLocal;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

public class NMFUpdate extends UpdateLocal<Tuple2<INDArray, INDArray>, Tuple2<INDArray, INDArray>> {
    double lower_bound;
    int omegaTest;
    static INDArray trainingDocument = null;
    
    public NMFUpdate(double lower_bound) {
        this.lower_bound = lower_bound;

    }


    @Override
    public Tuple2<INDArray, INDArray> process(Tuple2<INDArray, INDArray> input, ML4allContext context) {

        if(trainingDocument == null){
            trainingDocument = (INDArray)context.getByKey("testSet");
            this.omegaTest = this.calculateOmegavValue();
        }

        INDArray h = (INDArray) context.getByKey("h");
        INDArray w = (INDArray) context.getByKey("w");
        int i = (int)context.getByKey("i");
        int j = (int)context.getByKey("j");
        int currentItr = (int)context.getByKey("iter");
        int k = currentItr % 50;

        INDArray updateW = Transforms.max(w.getRow(i).sub(input.getField0()), lower_bound, true);
        INDArray updateH = Transforms.max(h.getColumn(j).sub(input.getField1()), lower_bound, true);

        w.putRow(i, updateW);
        h.putColumn(j, updateH);

        if(k == 0 || currentItr == 1) {
            double rmscValue = this.calcualteRMSE(i, j, w, h);
            System.out.println(rmscValue);
            context.put("rmsc", rmscValue);
        }

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


    private double calcualteRMSE(int row_i, int col_j, INDArray w, INDArray h) {

        double s = 0.0;
        for (int i = 0; i < this.trainingDocument.rows(); i++) {
            for (int j = 0; j < this.trainingDocument.columns(); j++) {
                if(this.trainingDocument.getDouble(row_i, col_j) > 0){
                    double s1 = this.trainingDocument.getDouble(row_i, col_j) - (w.getRow(row_i).mmul(h.getColumn(col_j))).getDouble(0);
                    s = s + Math.pow(s1, 2);
                }
            }
        }

        double s2 = s / omegaTest;

        return Math.sqrt(s2);
    }

    private int calculateOmegavValue(){
        int nonZeroValueCount = 0;
        for(int i =0; i < this.trainingDocument.rows(); i++){
            for(int j=0; j < this.trainingDocument.columns(); j++){
                if(this.trainingDocument.getDouble(i,j) > 0.0){
                    nonZeroValueCount++;
                }
            }
        }

        return nonZeroValueCount;
    }

}
