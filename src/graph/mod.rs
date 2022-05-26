#![allow(clippy::module_name_repetitions)]
// "factory" for building graphs
mod builder;
mod execute_engine;
mod graph;
mod operation;

pub use execute_engine::{Engine, ExecuteError};
pub use graph::{
    Graph, GraphError, GraphState, OperationId, OperationNode, PhysicalOp, RepresentationId,
};
pub use operation::Operation;
