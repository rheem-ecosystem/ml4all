package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.api.Transform;
import org.qcri.ml4all.examples.util.StringUtil;

import java.util.List;

public class NomadTransform extends Transform<INDArray, String> {

    char separator = ',';
    static int rowIndex = 0;
    static INDArray nd ;

    public NomadTransform(char separator, int m, int n) {

        this.separator = separator;
        this.nd = Nd4j.zeros(m,n);
    }

    @Override
    public INDArray transform(String input) {

        List<String> pointStr = StringUtil.split(input, separator);
        double[] point = new double[pointStr.size()];

        for(int i=0; i < pointStr.size(); i++){
            point[i] = Double.parseDouble(pointStr.get(i));
        }

        INDArray ndTemp = Nd4j.create(point, new int[]{1, pointStr.size()});
        nd.putRow(rowIndex, ndTemp);

        rowIndex++;

        return nd;
    }

}
