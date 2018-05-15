# ML4all: scalable ML system for everyone


ML4all is a system that frees users from the burden of machine learning algorithm selection and low-level implementation details.
It uses a new abstraction that is capable of solving most ML tasks and provides a cost-based optimizer on top of the proposed abstraction for choosing the best gradient descent algorithm in a given setting.
Our results show that ML4all is more than two orders of magnitude faster than state-of-the-art systems and can process large datasets that were not possible before.

More details can be found in our SIGMOD publication: [https://dl.acm.org/citation.cfm?id=3064042](https://dl.acm.org/citation.cfm?id=3064042)

Currently running on:
- Rheem 0.3.0
- Spark 1.6.x

Some examples can be found in org.qcri.ml4all.examples:
- kmeans: RunKmeans
- sgd: RunSGD