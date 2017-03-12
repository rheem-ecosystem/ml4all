package org.qcri.ml4all.abstraction.api;


import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class UpdateLocal<R, V> extends LogicalOperator {

    /**
     * Computes the new value of the global variable
     *
     * @param input the ouput of the aggregate of the {@link Compute}
     * @param context
     */
    public abstract R process(V input, ML4allContext context);

    /**
     * Assigns the new value of the global variable to the {@link ML4allContext}
     * @param input the output of the process method
     * @param context
     * @return the new {@link ML4allContext}
     */
    public abstract ML4allContext assign (R input, ML4allContext context); //TODO: deprecated class -> put input in a singleton list
}
