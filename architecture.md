# Library API (with some internal detail)

## GETTING STARTED

1. Input a network
2. Parse network into Operations
3. Construct a Graph from the Operations

## RUN NETWORK FORWARD WITH INPUT VECTORS

1. Get the Graph from GETTING STARTED
2. Construct an `ExecuteEngine` using the graph.
3. Construct inputs in the form of `Vec<(RepresentationId, Array1)>`, where the `Array1` value is a flattened tensor.
4. Call `run` on the engine instance, passing in the desired output `RepresentationId`s and a function that runs each operation on their inputs. An example is show below:
```
let res = engine.run(output_ids, inputs, |op, inputs, _| -> (Option<usize>, Vec<Array1>) {
    (None, op.forward1(inputs))
};
```

## Major definitions and abstractions

### Computation Graph

#### Representations

#### Operations

### Star

A representation of an input set and the linear transformation corresponding to a DNN's action on that set. The DNN transformation may be a prefix of a larger DNN.

### Starset

A directed graph of Stars with the following properties:

1. The starset has a root node defined by an input set and the identity transformation. This input set is also referred to as the Starset's input set.
2. Each edge corresponds to the action of a function that transforms a star into another star by incorporating the action of some DNN operation. The child star may have a different linear transformation from its parents' and must have an input set that's a subset of each its parents'.
3. DNN operations corresponding to edges must follow the partial order they are applied in the DNN. I.e., from the root, no path may visit edges out of order w.r.t. the order of the corresponding operations in the DNN.

### Branching operations

Branching operations are those that can divide a star's input set (e.g. ReLU) or is stochastic in nature (e.g. dropout). Each of these operations will correspond to at least 2 edges in the Starset graph from the parent star on which they are operating. The first class of branching operation are those that divide the input set. In this case, the children stars' input sets exclusive union is equal to the intersection of the input sets of the parent stars. In the second case of stochastic operations, the childrens' input sets are identical to the intersection of the of the parents' input sets.

### Prob StarSets

[TODO: talk about stochastic input variables and stochastic operations, as well as what can be calculated with this abstraction]

### Safe StarSets

[TODO: talk about ]

### Starset Tree

In the case that the DNN is entirely sequential, the Starset will take the form of a tree. Specifically, it will branch at certain operations 

### Starset Lattice

There is not a 1:1 relationship between starsets and DNN representations.

In a sequential model, because there is a single input and single output to every operation, we can work with a starset tree. There is a unique path from root starnode to leaf starnode because every operation must be run sequentially and we only need to worry about one representation at a time (i.e., once we update from R1 -> R2, R1 is obsolete).

However, in the graph model, we now need to think about branching networks. This means that the
