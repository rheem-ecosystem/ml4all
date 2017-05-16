package org.qcri.ml4all.examples.nomad;

import org.qcri.ml4all.abstraction.api.Transform;
import org.qcri.ml4all.examples.util.StringUtil;

import java.util.List;

public class NomadTransform extends Transform<double[], String> {

    char separator = ',';
    int features;

    public NomadTransform(int features, char separator) {
        this.features = features;
        this.separator = separator;
    }

    @Override
    public double[] transform(String line) {
        List<String> pointStr = StringUtil.split(line, separator);

        double [] point = new double[pointStr.size()];
        for (int i = 0; i < pointStr.size(); i++)
            point[i] = Double.parseDouble(pointStr.get(i));
        return point;
    }
}
