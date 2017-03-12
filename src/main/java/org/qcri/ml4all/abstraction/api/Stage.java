package org.qcri.ml4all.abstraction.api;

import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class Stage<R,V> extends LogicalOperator {

    /* initialize global variables based on the output of the {@link Transform} */
    public abstract R staging (V input, ML4allContext context);
}
