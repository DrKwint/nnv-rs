#![cfg(test)]
/// Performs tests on an x-structure graph, i.e. a graph with two inputs and two outputs.
use super::DummyOperation;
use crate::graph::{Engine, Graph, Operation, OperationId, OperationNode, RepresentationId};

/// Tests the following graph structure where letters indicate tensors and -> indicate operations:
/// Repr: A B C D E
/// ID:   0 1 2 3 4
/// (A, B) -> C: Operation 0
/// C -> D: Operation 1
/// E -> E: Operation 2
pub fn x_structure_graph() -> (Graph, Vec<RepresentationId>, Vec<usize>) {
    let mut graph = Graph::default();
    let repr_ids = (0..5)
        .map(|id| RepresentationId::new(id, None))
        .collect::<Vec<_>>();

    let ops = vec![(vec![0, 1], 2), (vec![2], 3), (vec![2], 4)]
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
fn test_x_structure_whole_graph() {
    let (graph, repr_ids, _) = x_structure_graph();
    let engine = Engine::new(&graph);
    let mut order = vec![];
    let run_res = engine.run(
        vec![repr_ids[3], repr_ids[4]],
        vec![(repr_ids[0], 0 as usize), (repr_ids[1], 0 as usize)],
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
fn test_x_structure_single_output() {
    let (graph, repr_ids, _) = x_structure_graph();
    let engine = Engine::new(&graph);
    let mut order = vec![];
    let run_res = engine.run(
        vec![repr_ids[4]],
        vec![(repr_ids[0], 0 as usize), (repr_ids[1], 0 as usize)],
        |operation: &(dyn Operation), _, _| -> (Option<usize>, Vec<usize>) {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            order.push(op.get_op_id());
            (None, vec![0])
        },
    );
    assert!(run_res.is_ok(), "{:?}", run_res);
    assert_eq!(order, vec![0 as OperationId, 2]);
}

#[test]
fn test_x_structure_invalid_subgraph() {
    let (graph, repr_ids, _) = x_structure_graph();
    let engine = Engine::new(&graph);
    // Test 3: Tests the subgraph (B -> C, C -> D)
    let mut order = vec![];
    let res = engine.run(
        vec![repr_ids[3]],
        vec![(repr_ids[0], 0 as usize)],
        |operation: &(dyn Operation), _, _| -> (Option<usize>, Vec<usize>) {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            order.push(op.get_op_id());
            (None, vec![0])
        },
    );
    assert!(res.is_err());
}
