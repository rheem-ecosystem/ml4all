package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.api.Transform;
import org.qcri.ml4all.examples.util.StringUtil;

import java.util.List;

public class NomadTransform extends Transform<INDArray, String> {

    char separator = ',';

    public NomadTransform(char separator) {
        this.separator = separator;
    }

    @Override
    public INDArray transform(String input) {

        List<String> pointStr = StringUtil.split(input, separator);
        double[] point = new double[pointStr.size()];
        for(int i=0; i < pointStr.size(); i++){
            point[i] = Double.parseDouble(pointStr.get(i));
        }

        INDArray nd = Nd4j.create(point, new int[]{1, pointStr.size()});

        return nd;
    }

}
