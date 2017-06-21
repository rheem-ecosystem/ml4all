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
    static int index_i = 0;

    public NMFTransform(char separator) {
        this.separator = separator;
    }

    @Override
    public double[] transform(String input) {
        List<String> pointStr = StringUtil.split(input, separator);

        //i,j,value
        double[] point = new double[pointStr.size() + 1];

        point[0] = index_i;

        for(int i=0; i < pointStr.size(); i++){
            point[i+1] = Double.parseDouble(pointStr.get(i));
        }

        index_i++;

        return point;
    }


}
