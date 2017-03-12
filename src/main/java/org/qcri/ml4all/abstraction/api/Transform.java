package org.qcri.ml4all.abstraction.api;

import java.io.Serializable;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class Transform<R, V> implements Serializable {

    /**
     * Parses and transforms an input data unit
     *
     * @param input usually a line of a file
     * @return the transformed data point
     */
    public abstract R transform (V input);

}
