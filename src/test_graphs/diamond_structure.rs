#![cfg(test)]
/// Performs tests on a diamond graph, i.e. a graph with one input to two nodes which are used to calculate an output.
use super::DummyOperation;
use crate::graph::{Engine, Graph, Operation, OperationId, OperationNode, RepresentationId};

/// Tests the following graph structure where letters indicate tensors and -> indicate operations:
/// Repr: A B C D
/// ID:   0 1 2 3
/// A -> B: Operation 0
/// A -> C: Operation 1
/// (B, C) -> D: Operation 2
pub fn diamond_structure_graph() -> (Graph, Vec<RepresentationId>, Vec<usize>) {
    let mut graph = Graph::default();
    let repr_ids = (0..4)
        .map(|id| RepresentationId::new(id, None))
        .collect::<Vec<_>>();

    let ops = vec![(vec![0], 1), (vec![0], 2), (vec![1, 2], 3)]
        .into_iter()
        .enumerate()
        .map(|(i, (input, output))| {
            OperationNode::new(
                Box::new(DummyOperation::new(i)),
                input.into_iter().map(|i| repr_ids[i]).collect::<Vec<_>>(),
                vec![repr_ids[output]],
            )
        })
        .collect::<Vec<_>>();

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

#[test]
fn test_diamond_structure_whole_graph() {
    let (graph, repr_ids, _) = diamond_structure_graph();
    let engine = Engine::new(&graph);
    let mut order = vec![];
    let run_res = engine.run(
        vec![repr_ids[3]],
        &vec![(repr_ids[0], 0 as usize)],
        |operation: &dyn Operation, _, _| -> (Option<usize>, Vec<usize>) {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            order.push(op.get_op_id());
            (None, vec![0])
        },
    );
    assert!(run_res.is_ok(), "{:?}", run_res);
    assert_eq!(order, vec![0 as OperationId, 1, 2]);
}

#[test]
fn test_diamond_structure_subgraph_output() {
    let (graph, repr_ids, _) = diamond_structure_graph();
    let engine = Engine::new(&graph);
    let mut order = vec![];
    let run_res = engine.run(
        vec![repr_ids[2]],
        &vec![(repr_ids[0], 0 as usize)],
        |operation: &(dyn Operation), _, _| -> (Option<usize>, Vec<usize>) {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            order.push(op.get_op_id());
            (None, vec![0])
        },
    );
    assert!(run_res.is_ok(), "{:?}", run_res);
    assert_eq!(order, vec![1 as OperationId]);
}
