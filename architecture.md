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

## Assumptions to sort:

Parent star not empty. Empty stars are not added to a starset.

## Major definitions and abstractions

### DeepPoly

### Computation Graph

#### Representations

#### Operations

#### Bounds

Input Bounds
Output Bounds
Local Output Bounds

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

Starsets can be interpreted in a probabilistic way. Given a stochastic input variable, we can calculate the likelihood of an output by calculating the likelihood that an input sample falls into the corresponding input set. The other source of stochasticity is the set of probabilistic operations in the DNN (e.g., dropout).

### Safe StarSets

[TODO]

### DNN and Starset structure

Let's consider the relationship between DNN representations and Starset stars. At the input representation, there's a 1:1 relationship, but as branching operations are applied, the number of DNN represetations stays constant while the number of stars grows exponentially. Generally, the starset is arranged into a tree, where there is a unique path from the root star to any star in the set. Consider the following diamond DNN where each tensor is of shape `()`:

```Python
w = tf.keras.Input()
x = tf.keras.Linear(w)
y = tf.keras.Linear(w)
z = x + y
```

The operation graph forms a simple diamond, but the star set will be a little more sophisticated. Let's consider the set of valid representation sets: `{w}, {wx}, {wy}, {xy}, {z}`. This is pretty straightforward, but that's only because the DNN doesn't have any branching operations. Let's now consider the following DNN that adds stepReLUs:

```Python
a = tf.keras.Input()
b = tf.keras.Linear(a)
c = tf.keras.ReLU(b)
d = tf.keras.Linear(a)
e = tf.keras.ReLU(d)
f = c + e
```

This network is only slightly more complex in terms of the operation graph, but let's think about what the star set looks like now. The 'H's and 'L's correspond to high and low outputs from the stepReLU operation.

![Starset])(/images/graphviz.svg)

We notice that there are many stars that correspond to the representation produced by a sequence of operations. Specifically, representation `f` has 4 stars corresponding to it. Note, as well, that the input set of some of the stars might be empty. Consider the constraints on `a` included in the `c` and `e` stars. WLOG, assume the `c` threshold is less than the `e` threshold, so the input set of `a` is partitioned into 3 groups. By the pidgeonhole principle, because each `f` star's input set is an intersection of the parent stars', one of the `f` stars must be empty. `f1` and `f4` will have non-empty inputs because they correspond to both `c` and `e` being low and high respectively. The middle set corresponds to the input set of `f3`, the intersection of the `ch` and `el` stars. Thus, `f2`'s input set, the intersection of `cl` and `eh` is empty. In practice, we check whether each star is empty before adding it to the starset, excluding it if that's the case.