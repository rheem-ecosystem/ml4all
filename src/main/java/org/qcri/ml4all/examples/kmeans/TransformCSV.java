package org.qcri.ml4all.examples.kmeans;

import org.qcri.ml4all.abstraction.api.Transform;
import org.qcri.ml4all.examples.util.StringUtil;

import java.util.List;

/**
 * Created by zoi on 22/1/15.
 */
public class TransformCSV extends Transform<double[], String> {

    char separator = ',';

    public TransformCSV () {

    }

    public TransformCSV(char separator) {
        this.separator = separator;
    }

    @Override
    public double[] transform(String input) {
        List<String> pointStr = StringUtil.split(input, separator);
        double [] point = new double[pointStr.size()];
        for (int i = 0; i < pointStr.size(); i++)
            point[i] = Double.parseDouble(pointStr.get(i));
        return point;
    }

}
