package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Compute;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zoi on 25/1/15.
 */
public class ComputePerPartitionWrapper<R, V> extends LogicalOperatorWrapperWithContext<Iterable<R>, Iterable<V>> {

    Compute<R, V> logOp;

    R sumOfPartition;

    public ComputePerPartitionWrapper(Compute logOp) {
        this.logOp = logOp;
    }

    @Override
    public Iterable<R> apply(Iterable<V> o) {
        List<R> list = new ArrayList<>(1);
        o.forEach(p -> sumOfPartition = this.logOp.process(p, context));
        list.add(sumOfPartition);
        return list;
    }

    @Override
    public void initialise() { logOp.initialise(); }

}
