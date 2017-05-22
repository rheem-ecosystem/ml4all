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

    public NMFTransform(char separator) {
        this.separator = separator;
    }

    @Override
    public double[] transform(String input) {
        // assume it has id

        List<String> pointStr = StringUtil.split(input, separator);
        double[] point = new double[pointStr.size()];

        // to get i value. will be replaced when we know input file format
        point[0] = Double.parseDouble(String.valueOf(pointStr.get(0).charAt(pointStr.get(0).length() - 1)));

        for(int i=1; i < pointStr.size(); i++){
            point[i] = Double.parseDouble(pointStr.get(i));
        }

        return point;
    }

}
