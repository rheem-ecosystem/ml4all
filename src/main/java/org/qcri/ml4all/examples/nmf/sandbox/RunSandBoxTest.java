package org.qcri.ml4all.examples.nmf.sandbox;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.net.MalformedURLException;

import static org.qcri.ml4all.abstraction.plan.Platforms.*;

/**
 * Execute SGD for logistic regression.
 */
public class RunSandBoxTest {

    public static void main (String... args) throws MalformedURLException {

        new NMFJuliaBatchMatrixFactorization();
    }

}


