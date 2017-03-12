package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Transform;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zoi on 25/1/15.
 */
public class TransformPerPartitionWrapper<R, V> extends LogicalOperatorWrapper<Iterable<R>, Iterable<V>> {

    Transform<R, V> logOp;

    public TransformPerPartitionWrapper(Transform logOp) {
        this.logOp = logOp;
    }

    @Override
    public Iterable<R> apply(Iterable<V> o) {
        List<R> list = new ArrayList<>();
        o.forEach(p -> list.add(this.logOp.transform(p)));
        return list;
    }
}
