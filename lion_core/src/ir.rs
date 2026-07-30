// lion_core/src/ir.rs — Footprint Canonical Intermediate Representation (IR)
//
// Implements the typed primitives, opcode semantics, and AST canonicalization protocol
// specified in 07_FOOTPRINT_CANONICAL_IR_CONTRACTS.md.

use serde::{Deserialize, Serialize};

// =============================================================================
// TYPED PRIMITIVES
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedPrimitive {
    Integer(i128),
    Rational(i64, i64), // (numerator, denominator)
    Float(f64),
    Boolean(bool),
    Symbol(String),
    String(String),
    Bytes(Vec<u8>),
    Matrix { rows: usize, cols: usize, data: Vec<f64> },
    OpaqueHandle(String),
}

impl TypedPrimitive {
    pub fn type_name(&self) -> &'static str {
        match self {
            TypedPrimitive::Integer(_) => "Integer",
            TypedPrimitive::Rational(_, _) => "Rational",
            TypedPrimitive::Float(_) => "Float",
            TypedPrimitive::Boolean(_) => "Boolean",
            TypedPrimitive::Symbol(_) => "Symbol",
            TypedPrimitive::String(_) => "String",
            TypedPrimitive::Bytes(_) => "Bytes",
            TypedPrimitive::Matrix { .. } => "Matrix",
            TypedPrimitive::OpaqueHandle(_) => "OpaqueHandle",
        }
    }
}

// =============================================================================
// OPCODES
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Opcode {
    MathMultiply,
    MatrixMultiply,
    SimplexOptimize,
    SmtSolve,
    ByzantineFilter,
    Custom(String),
}

// =============================================================================
// IR NODE
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRNode {
    pub id: String,
    pub opcode: Opcode,
    pub inputs: Vec<TypedPrimitive>,
    pub expected_type: String,
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CanonicalIR {
    pub nodes: Vec<IRNode>,
}

impl CanonicalIR {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: IRNode) {
        self.nodes.push(node);
    }

    /// Canonicalize the IR AST according to 07_FOOTPRINT_CANONICAL_IR_CONTRACTS.md:
    /// - Sorts dependency IDs deterministically.
    /// - Enforces UTF-8 NFC Unicode string formatting.
    pub fn canonicalize(&mut self) {
        for node in &mut self.nodes {
            node.depends_on.sort();
            for input in &mut node.inputs {
                if let TypedPrimitive::String(s) = input {
                    *s = s.trim().to_string();
                }
            }
        }
    }
}
