// "factory" for building graphs
mod builder;
mod deeppoly;
mod execute_engine;
mod graph;
mod operation;

pub use execute_engine::Engine;
pub use graph::{Graph, GraphError, OperationId, RepresentationId};
pub(crate) use operation::Operation;
