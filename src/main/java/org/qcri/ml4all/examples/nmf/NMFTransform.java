package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.api.Transform;
import org.qcri.ml4all.examples.util.StringUtil;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NMFTransform extends Transform<double[] , String> {

    char separator = ',';
    int index_i = 0;
    public NMFTransform(char separator) {

        this.separator = separator;
    }

    @Override
    public double[] transform(String input) {
        List<String> pointStr = StringUtil.split(input, separator);

        //i,j,value
        double[] point = new double[3];

        point[0] = index_i;
        int j = this.getRandomIndexPointer(0, pointStr.size() -1);
        point[1] = j;
        point[2] = Double.parseDouble(pointStr.get(j));
        index_i++;
        return point;
    }

    private int getRandomIndexPointer(int min, int max){
        return min+(int)(Math.random()*((max-min ) + 1));
    }

}
