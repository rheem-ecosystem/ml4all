package org.qcri.ml4all.examples.nmf.sandbox;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by jlucas on 6/12/17.
 */
public class APGFeature {

    public int count;
    public double sum;
    public Map<Integer, Double> nonZeros;

    public APGFeature(double sum, int count,  Map<Integer, Double> nonZeros) {
        this.sum = sum;
        this.count = count;
        this.nonZeros = nonZeros;
    }
}
