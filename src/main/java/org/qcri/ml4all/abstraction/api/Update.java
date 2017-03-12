package org.qcri.ml4all.abstraction.api;


import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.util.List;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class Update<R, V> extends LogicalOperator {

    /**
     * Computes the new value of the global variable
     *
     * @param input the ouput of the aggregate of the {@link Compute}
     * @param context
     */
    public abstract R process(V input, ML4allContext context);

    /**
     * Assigns the new value of the global variable to the {@link ML4allContext}
     * @param input a list of all outputs of the process method
     * @param context
     * @return the new {@link ML4allContext}
     */
    public abstract ML4allContext assign (List<R> input, ML4allContext context);
}
