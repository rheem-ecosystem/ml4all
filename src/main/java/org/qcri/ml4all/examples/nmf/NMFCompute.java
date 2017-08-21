package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.sql.Timestamp;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class NMFCompute extends Compute<Tuple2<Tuple2, int[]>, double[]> {

    private double beta =  0.00;
    private int datasetSize;
    private int feasureSize;
    private INDArray trainingDocument;

    public NMFCompute(double beta, int datasetSize, int featureSize, INDArray trainingDocument) {
        this.beta = beta;
        this.datasetSize = datasetSize;
        this.feasureSize = featureSize;
        this.trainingDocument =  trainingDocument;
    }

    @Override
    public Tuple2 process(double[] input , ML4allContext context) {
        INDArray updateW = null;
        INDArray updateH = null;


        int i = this.getRandomIndexPointer(0, this.datasetSize -1);
        int j = this.getRandomIndexPointer(0, this.feasureSize - 1);

        int[] point = new int[]{i,j};

        double aDataPoint = this.trainingDocument.getDouble(i,j);


        double alpha = (double)context.getByKey("alpha");
        INDArray w = (INDArray)context.getByKey("w");
        INDArray h = (INDArray)context.getByKey("h");


        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));


        updateW = ((h.getColumn(j).mul(aW)).sub(w.getRow(i).mul(this.beta))).mul(alpha).transpose();

        updateH = ((w.getRow(i).mul(aW)).sub(h.getColumn(j).mul(this.beta))).mul(alpha).transpose();

        return new Tuple2(new Tuple2(updateW, updateH), new Tuple2(point, aW));
    }


    private int getRandomIndexPointer(int min, int max){
        return min+(int)(Math.random()*((max-min ) + 1));
    }

}
