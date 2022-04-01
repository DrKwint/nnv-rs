#![cfg(test)]
/// Performs tests on a y-structure graph, i.e. a graph with one input and two outputs.
use super::DummyOperation;
use crate::graph::{Engine, Graph, Operation, OperationId, OperationNode, RepresentationId};

/// Tests the following graph structure where letters indicate tensors and -> indicate operations:
/// Repr: A B C D E F
/// ID:   0 1 2 3 4 5
/// A -> B: Operation 0
/// B -> C: Operation 1
/// C -> D: Operation 2
/// B -> E: Operation 3
/// E -> F: Operation 4
pub fn y_structure_graph() -> (Graph, Vec<RepresentationId>, Vec<usize>) {
    let mut graph = Graph::default();
    let repr_ids = (0..6)
        .map(|id| RepresentationId::new(id, None))
        .collect::<Vec<_>>();

    let ops = vec![(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
        .into_iter()
        .enumerate()
        .map(|(i, (input, output))| {
            OperationNode::new(
                Box::new(DummyOperation::new(i)),
                vec![repr_ids[input]],
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
fn test_y_structure_graph_whole_graph() {
    let (graph, repr_ids, _) = y_structure_graph();
    let engine = Engine::new(&graph);
    let mut order = vec![];
    let run_res = engine.run(
        vec![repr_ids[3], repr_ids[5]],
        &vec![(repr_ids[0], 0 as usize)],
        |operation: &dyn Operation, _, _| -> (Option<usize>, Vec<usize>) {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            order.push(op.get_op_id());
            (None, vec![0])
        },
    );
    assert!(run_res.is_ok(), "{:?}", run_res);
    assert_eq!(order, vec![0 as OperationId, 1, 2, 3, 4]);
}

#[test]
fn test_y_structure_subgraph() {
    let (graph, repr_ids, _) = y_structure_graph();
    let engine = Engine::new(&graph);

    let mut order = vec![];
    let run_res = engine.run(
        vec![repr_ids[2], repr_ids[4]],
        &vec![(repr_ids[0], 0 as usize)],
        |operation: &(dyn Operation), _, _| -> (Option<usize>, Vec<usize>) {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            order.push(op.get_op_id());
            (None, vec![0])
        },
    );
    assert!(run_res.is_ok(), "{:?}", run_res);
    assert_eq!(order, vec![0 as OperationId, 1, 3]);
}

#[test]
fn test_y_structure_short_sub() {
    let (graph, repr_ids, _) = y_structure_graph();
    let engine = Engine::new(&graph);
    let mut order = vec![];
    let run_res = engine.run(
        vec![repr_ids[3]],
        &vec![(repr_ids[1], 0 as usize)],
        |operation: &(dyn Operation), _, _| -> (Option<usize>, Vec<usize>) {
            let op = operation.as_any().downcast_ref::<DummyOperation>().unwrap();
            order.push(op.get_op_id());
            (None, vec![0])
        },
    );
    assert!(run_res.is_ok(), "{:?}", run_res);
    assert_eq!(order, vec![1 as OperationId, 2]);
}
