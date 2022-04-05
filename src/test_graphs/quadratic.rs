#![cfg(test)]
/// Computes values of f(x) = ax^2 + bx + c using graph computation
use super::{SimpleAdd, SimpleMultiply, SimpleSquare};
use crate::graph::{Engine, Graph, Operation, OperationNode, PhysicalOp, RepresentationId};
use crate::test_util::*;
use crate::NNVFloat;
use ndarray::Array1;
use proptest::prelude::*;

/// Tests the following graph structure where letters indicate tensors and -> indicate operations:
/// Repr: x a b c x^2 ax^2 bx ax^2+bx ax^2+bx+c
/// ID:   0 1 2 3 4   5    6  7       8
/// x            -> x^2:       0      -> 4: Operation 0
/// (a, x^2)     -> ax^2:      (1, 4) -> 5: Operation 1
/// (b, x)       -> bx:        (2, 0) -> 6: Operation 2
/// (ax^2, bx)   -> ax^2+bx:   (5, 6) -> 7: Operation 3
/// (ax^2+bx, c) -> ax^2+bx+c: (7, 3) -> 8: Operation 4
pub fn quadratic_graph() -> (Graph, Vec<RepresentationId>, Vec<usize>) {
    let mut graph = Graph::default();
    let repr_ids = (0..9)
        .map(|id| RepresentationId::new(id, None))
        .collect::<Vec<_>>();

    let ops = vec![
        // x -> x^2
        OperationNode::new(
            PhysicalOp::from(SimpleSquare::default()),
            vec![repr_ids[0]],
            vec![repr_ids[4]],
        ),
        // (a, x^2) -> ax^2
        OperationNode::new(
            PhysicalOp::from(SimpleMultiply::default()),
            vec![repr_ids[1], repr_ids[4]],
            vec![repr_ids[5]],
        ),
        // (b, x) -> bx
        OperationNode::new(
            PhysicalOp::from(SimpleMultiply::default()),
            vec![repr_ids[2], repr_ids[0]],
            vec![repr_ids[6]],
        ),
        // (ax^2, bx) -> ax^2+bx
        OperationNode::new(
            PhysicalOp::from(SimpleAdd::default()),
            vec![repr_ids[5], repr_ids[6]],
            vec![repr_ids[7]],
        ),
        // (ax^2+bx, c) -> ax^2+bx+c
        OperationNode::new(
            PhysicalOp::from(SimpleAdd::default()),
            vec![repr_ids[7], repr_ids[3]],
            vec![repr_ids[8]],
        ),
    ];

    let op_ids = ops
        .into_iter()
        .map(|op| graph.add_operation_node(op))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    for (i, op_id) in op_ids.iter().enumerate() {
        assert_eq!(i, *op_id);
    }

    (graph, repr_ids, op_ids)
}

proptest! {
    #[test]
    fn test_quadratic(a in array1(5), b in array1(5), c in array1(5), x in array1(5)) {
        let (graph, repr_ids, _) = quadratic_graph();
        let engine = Engine::new(&graph);
        let run_res = engine.run(
            vec![repr_ids[8]],
            &vec![(repr_ids[0], x.clone()), (repr_ids[1], a.clone()), (repr_ids[2], b.clone()), (repr_ids[3], c.clone())],
            |operation: &PhysicalOp, input, _| -> (Option<usize>, Vec<Array1<NNVFloat>>) {
                let output = operation.forward1(input);
                (None, output)
            },
        );
        prop_assert!(run_res.is_ok(), "{:?}", run_res);
        let graph_outputs = run_res.unwrap();
        prop_assert_eq!(graph_outputs.len(), 1);

        let output = a * x.clone() * x.clone() + b * x + c;
        prop_assert!(graph_outputs[0].1.abs_diff_eq(&output, 1e-8));
    }
}
