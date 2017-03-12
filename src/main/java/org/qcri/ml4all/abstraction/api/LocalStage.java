package org.qcri.ml4all.abstraction.api;

import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class LocalStage extends LogicalOperator {

    /* initialize variables and add them in the context */
    public abstract void staging (ML4allContext context);
}
