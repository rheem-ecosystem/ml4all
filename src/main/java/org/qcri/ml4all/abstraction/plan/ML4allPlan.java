package org.qcri.ml4all.abstraction.plan;


import gnu.trove.map.hash.THashMap;
import org.qcri.ml4all.abstraction.api.*;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.ml4all.abstraction.plan.wrappers.*;
import org.qcri.rheem.api.DataQuantaBuilder;
import org.qcri.rheem.api.JavaPlanBuilder;
import org.qcri.rheem.basic.data.Tuple2;
import org.qcri.rheem.basic.operators.SampleOperator;
import org.qcri.rheem.core.api.Configuration;
import org.qcri.rheem.core.api.RheemContext;
import org.qcri.rheem.core.function.PredicateDescriptor;
import org.qcri.rheem.core.util.ReflectionUtils;
import org.qcri.rheem.core.util.RheemCollections;
import org.qcri.rheem.core.util.Tuple;
import org.qcri.rheem.java.Java;
import org.qcri.rheem.java.platform.JavaPlatform;
import org.qcri.rheem.spark.Spark;

import java.util.ArrayList;
import java.util.Collection;

//import org.qcri.rheem.java.Java;
//import org.qcri.rheem.spark.Spark;

/**
 * Rheem physical plan for ML4all algorithms
 */

public class ML4allPlan {

    Transform transformOp;
    Class transformIn = String.class; //default value: String
    Class transformOut = double[].class; //default value: double[]

    LocalStage localStageOp;

    Compute computeOp;
    Class computeOutKey = Integer.class; //default: integer
    Class computeOutValue = double[].class; //default: double[]

    Update updateOp;
    UpdateLocal updateLocalOp;

    Loop loopOp;
    Class loopConvergeOut;

    Sample sampleOp;

    long datasetsize;

    public void setTransformOp(Transform transformOp) {
        this.transformOp = transformOp;
    }

    public void setTransformInput(Class transformIn) {
        this.transformIn = transformIn;
    }

    public void setTransformOutput(Class transformOut) {
        this.transformOut = transformOut;
    }

    public void setLocalStage(LocalStage stageOp) {
        this.localStageOp = stageOp;
    }

    public void setComputeOp(Compute computeOp) {
        this.computeOp = computeOp;
    }

    public void setComputeOutput(Class key, Class value) {
        this.computeOutKey = key;
        this.computeOutValue = value;
    }

    public void setSampleOp(Sample sampleOp) {
        this.sampleOp = sampleOp;
    }

    public void setUpdateOp(Update updateOp) {
        this.updateOp = updateOp;
    }

    public void setUpdateLocalOp(UpdateLocal updateLocalOp) {
        this.updateLocalOp = updateLocalOp;
    }

    public void setLoopOp(Loop loopOp) { this.loopOp = loopOp; }

    public void setLoopConvergeOutput(Class loopConvergeOut) { this.loopConvergeOut = loopConvergeOut; }

    public boolean isUpdateLocal() {
        return updateLocalOp != null;
    }

    public boolean hasSample() { return sampleOp != null; }

    public void setDatasetsize(long datasetsize) {
        this.datasetsize = datasetsize;
    }


    public RheemContext initiateRheemContext(Platforms platform, String propertiesFile) {
        // Instantiate Rheem and activate the backend.
        Configuration conf;
        if (propertiesFile == null)
            conf = new Configuration();
        else
            conf = new Configuration(propertiesFile);
        RheemContext rheemContext = new RheemContext(conf);

        switch (platform) {
            case SPARK:
                rheemContext.with(Spark.basicPlugin());
                break;
            case JAVA:
                rheemContext.with(Java.basicPlugin());
                break;
            case SPARK_JAVA:
                rheemContext.with(Java.basicPlugin());
                rheemContext.with(Spark.basicPlugin());
                break;
            default:
                System.err.format("Unknown platform: \"%s\"\n", platform);
                System.exit(3);
                return null;
        }
        rheemContext.getConfiguration().setProperty("rheem.core.optimizer.reoptimize", "false");
        return rheemContext;
    }

    /*
     * Return the last state of ML4allContext that contains the model
    */
    public ML4allContext execute(String inputFileUrl, Platforms platform, String propertiesFile) {
        // Instantiate Rheem and activate the backend.
        RheemContext rheemContext = initiateRheemContext(platform, propertiesFile);

        JavaPlanBuilder javaPlanBuilder = new JavaPlanBuilder(rheemContext)
                .withUdfJar(ReflectionUtils.getDeclaringJar(ML4allContext.class))
                .withUdfJar(ReflectionUtils.getDeclaringJar(JavaPlatform.class))
                .withUdfJar(ReflectionUtils.getDeclaringJar(THashMap.class))
                .withJobName("ML4all plan");

        ML4allContext context = new ML4allContext();
        localStageOp.staging(context);
        ArrayList<ML4allContext> broadcastContext = new ArrayList<>(1);
        broadcastContext.add(context);
        final DataQuantaBuilder<?, ML4allContext> contextBuilder = javaPlanBuilder.loadCollection(broadcastContext).withName("init context");

        if (platform.equals(Platforms.SPARK_JAVA)) {
            final DataQuantaBuilder transformBuilder = javaPlanBuilder
                    .readTextFile(inputFileUrl).withName("source")
                    .mapPartitions(new TransformPerPartitionWrapper(transformOp)).withName("transform")
                    .withTargetPlatform(Spark.platform());

            Collection<ML4allContext> results =
                    contextBuilder.doWhile((PredicateDescriptor.SerializablePredicate<Collection<Double>>) collection ->
                            new LoopCheckWrapper<>(loopOp).apply(collection.iterator().next()), ctx -> {

                        DataQuantaBuilder convergenceDataset; //TODO: don't restrict the convergence value to be double
                        DataQuantaBuilder<?, ML4allContext> newContext;

                        DataQuantaBuilder sampledData;
                        if (hasSample()) //sample data first
                            sampledData = transformBuilder
                                    .sample(sampleOp.sampleSize()).withSampleMethod(sampleOp.sampleMethod()).withDatasetSize(datasetsize).withBroadcast(ctx, "context").withTargetPlatform(Spark.platform());
                        else //sampled data is entire dataset
                            sampledData = transformBuilder;

                        if (isUpdateLocal()) { //eg., for GD
                            DataQuantaBuilder newWeights = sampledData
                                    .map(new ComputeWrapper<>(computeOp)).withBroadcast(ctx, "context").withName("compute")
                                    .reduce(new AggregateWrapper<>(computeOp)).withName("reduce")
                                    .map(new UpdateLocalWrapper(updateLocalOp)).withBroadcast(ctx, "context").withName("update").withTargetPlatform(Java.platform());

                            newContext = newWeights
                                    .map(new AssignWrapperLocal(updateLocalOp)).withName("assign")
                                    .withBroadcast(ctx, "context")
                                    .withTargetPlatform(Java.platform());
                            convergenceDataset = newWeights
                                    .map(new LoopConvergenceWrapper(loopOp)).withName("converge")
                                    .withBroadcast(ctx, "context")
                                    .withTargetPlatform(Java.platform());
                        } else { //eg., for k-means
                            DataQuantaBuilder listDataset = sampledData
                                    .map(new ComputeWrapper<>(computeOp)).withBroadcast(ctx, "context").withName("compute")
                                    .reduceByKey(pair -> ((Tuple2) pair).field0, new AggregateWrapper<>(computeOp)).withName("reduce")
                                    .map(new UpdateWrapper(updateOp)).withBroadcast(ctx, "context").withName("update")
                                    .map(t -> {
                                        ArrayList<Tuple2> list = new ArrayList<>(1);
                                        list.add((Tuple2) t);
                                        return list;
                                    })
                                    .reduce(new ReduceWrapper<>()).withName("global reduce")
                                    .withTargetPlatform(Spark.platform());
                            newContext = listDataset
                                    .map(new AssignWrapper(updateOp)).withName("assign")
                                    .withBroadcast(ctx, "context")
                                    .withTargetPlatform(Java.platform());
                            convergenceDataset = listDataset
                                    .map(new LoopConvergenceWrapper(loopOp)).withName("converge")
                                    .withBroadcast(ctx, "context")
                                    .withTargetPlatform(Java.platform());
                        }

                        return new Tuple<>(newContext, convergenceDataset);
                    }).withTargetPlatform(Java.platform())
                      .collect();
            return RheemCollections.getSingle(results);
        }
        else {
            final DataQuantaBuilder transformBuilder = javaPlanBuilder
                    .readTextFile(inputFileUrl).withName("source")
                    .mapPartitions(new TransformPerPartitionWrapper(transformOp)).withName("transform");

            Collection<ML4allContext> results =
                    contextBuilder.doWhile((PredicateDescriptor.SerializablePredicate<Collection<Double>>) collection ->
                            new LoopCheckWrapper<>(loopOp).apply(collection.iterator().next()), ctx -> {

                        DataQuantaBuilder convergenceDataset;
                        DataQuantaBuilder<?, ML4allContext> newContext;

                        DataQuantaBuilder sampledData;
                        if (hasSample()) //sample data first
                            sampledData = transformBuilder
                                    .sample(sampleOp.sampleSize()).withSampleMethod(sampleOp.sampleMethod()).withDatasetSize(datasetsize).withBroadcast(ctx, "context");
                        else //sampled data is entire dataset
                            sampledData = transformBuilder;

                        if (isUpdateLocal()) { //eg., for GD
                            DataQuantaBuilder newWeights = sampledData
                                    .map(new ComputeWrapper<>(computeOp)).withBroadcast(ctx, "context").withName("compute")
                                    .reduce(new AggregateWrapper<>(computeOp)).withName("reduce")
                                    .map(new UpdateLocalWrapper(updateLocalOp)).withBroadcast(ctx, "context").withName("update");

                            newContext = newWeights
                                    .map(new AssignWrapperLocal(updateLocalOp)).withName("assign")
                                    .withBroadcast(ctx, "context");

                            convergenceDataset = newWeights
                                    .map(new LoopConvergenceWrapper(loopOp)).withName("converge")
                                    .withBroadcast(ctx, "context");

                        } else { //eg., for k-means
                            DataQuantaBuilder listDataset = sampledData
                                    .map(new ComputeWrapper<>(computeOp)).withBroadcast(ctx, "context").withName("compute")
                                    .reduceByKey(pair -> ((Tuple2) pair).field0, new AggregateWrapper<>(computeOp)).withName("reduce")
                                    .map(new UpdateWrapper(updateOp)).withBroadcast(ctx, "context").withName("update")
                                    .map(t -> {
                                        ArrayList<Tuple2> list = new ArrayList<>(1);
                                        list.add((Tuple2) t);
                                        return list;
                                    })
                                    .reduce(new ReduceWrapper<>()).withName("global reduce");
                            newContext = listDataset
                                    .map(new AssignWrapper(updateOp)).withName("assign")
                                    .withBroadcast(ctx, "context");
                            convergenceDataset = listDataset
                                    .map(new LoopConvergenceWrapper(loopOp)).withName("converge")
                                    .withBroadcast(ctx, "context");
                        }

                        return new Tuple<>(newContext, convergenceDataset);
                    }).collect();

            return RheemCollections.getSingle(results);
        }
    }

}

