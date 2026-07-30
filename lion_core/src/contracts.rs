// lion_core/src/contracts.rs — Footprint Semantic Verification Contracts
//
// Implements Design by Contract static verification from 07_FOOTPRINT_CANONICAL_IR_CONTRACTS.md.

use serde::{Deserialize, Serialize};
use crate::ir::{IRNode, Opcode, TypedPrimitive};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationContract {
    pub opcode: Opcode,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub memory_ceiling_mb: usize,
}

pub struct SemanticAnalyzer;

impl SemanticAnalyzer {
    /// Statically verify IR node against preconditions before execution.
    pub fn verify_node(node: &IRNode) -> Result<(), String> {
        match &node.opcode {
            Opcode::MatrixMultiply => {
                if node.inputs.len() < 2 {
                    return Err("MatrixMultiply requires exactly 2 input matrices".to_string());
                }
                match (&node.inputs[0], &node.inputs[1]) {
                    (
                        TypedPrimitive::Matrix { cols: c1, .. },
                        TypedPrimitive::Matrix { rows: r2, .. },
                    ) => {
                        if c1 != r2 {
                            return Err(format!(
                                "Static Contract Violation: Matrix A cols ({}) != Matrix B rows ({})",
                                c1, r2
                            ));
                        }
                    }
                    _ => return Err("Inputs must be of type Matrix".to_string()),
                }
            }
            Opcode::MathMultiply => {
                if node.inputs.len() < 2 {
                    return Err("MathMultiply requires at least 2 numerical inputs".to_string());
                }
            }
            _ => {}
        }
        Ok(())
    }
}
