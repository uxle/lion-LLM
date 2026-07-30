// lion_agent/src/runtime/dag.rs — DAG Scheduler (Topological Batching)
//
// Invariant: DEPENDENCIES_DAG_VALIDATED
// The DAG is computed once, before any execution begins.
// Dependencies are never inferred at execution time.

use std::collections::HashSet;
use crate::runtime::types::PlanAction;
use crate::runtime::error::{RuntimeError, RuntimeResult};

/// Partition `actions` into sequential batches where every action in a batch
/// has all its dependencies satisfied by earlier batches.
///
/// Returns Vec<Vec<PlanAction>> — each inner Vec runs in parallel.
/// Errors on cyclic or referentially invalid dependency graphs.
///
/// Example:
///   A → {B, C} → D
///   Batch 1: [A]
///   Batch 2: [B, C]   ← parallel
///   Batch 3: [D]
pub fn topologically_batch(actions: Vec<PlanAction>) -> RuntimeResult<Vec<Vec<PlanAction>>> {
    // Verify all dependsOn references are valid action IDs
    let all_ids: HashSet<&str> = actions.iter().map(|a| a.id.as_str()).collect();
    for action in &actions {
        for dep in &action.depends_on {
            if !all_ids.contains(dep.as_str()) {
                return Err(RuntimeError::CyclicDependencyGraph);
            }
        }
    }

    let mut remaining: Vec<PlanAction> = actions;
    let mut completed: HashSet<String>  = HashSet::new();
    let mut batches:   Vec<Vec<PlanAction>> = Vec::new();

    // Kahn's algorithm: iteratively peel off actions whose deps are all done.
    while !remaining.is_empty() {
        let (ready, not_ready): (Vec<PlanAction>, Vec<PlanAction>) = remaining
            .into_iter()
            .partition(|action| {
                action.depends_on.iter().all(|dep| completed.contains(dep))
            });

        if ready.is_empty() {
            // No progress made — must be a cycle
            return Err(RuntimeError::CyclicDependencyGraph);
        }

        // Mark this batch's actions as complete before next iteration
        for action in &ready {
            completed.insert(action.id.clone());
        }

        batches.push(ready);
        remaining = not_ready;
    }

    Ok(batches)
}

/// Validate that a plan has no structural issues before scheduling.
pub fn validate_plan_dag(actions: &[PlanAction]) -> Result<(), String> {
    let all_ids: HashSet<&str> = actions.iter().map(|a| a.id.as_str()).collect();

    // Check for duplicate IDs
    if all_ids.len() != actions.len() {
        return Err("Duplicate action IDs in plan".to_string());
    }

    // Check all dep references are valid
    for action in actions {
        for dep in &action.depends_on {
            if !all_ids.contains(dep.as_str()) {
                return Err(format!(
                    "Action '{}' depends on unknown action '{}'", action.id, dep
                ));
            }
            // Self-dependency
            if dep == &action.id {
                return Err(format!("Action '{}' depends on itself", action.id));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::types::{RetryPolicy, RiskLevel};

    fn action(id: &str, deps: &[&str]) -> PlanAction {
        PlanAction {
            id:                    id.to_string(),
            tool_id:               "echo".to_string(),
            arguments:             Default::default(),
            depends_on:            deps.iter().map(|s| s.to_string()).collect(),
            risk_level:            RiskLevel::Low,
            requires_authorization: false,
            timeout_ms:            1000,
            retry_policy:          RetryPolicy::default(),
        }
    }

    #[test]
    fn test_linear_chain() {
        // A → B → C
        let batches = topologically_batch(vec![
            action("a", &[]),
            action("b", &["a"]),
            action("c", &["b"]),
        ]).unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].id, "a");
        assert_eq!(batches[1][0].id, "b");
        assert_eq!(batches[2][0].id, "c");
    }

    #[test]
    fn test_diamond_dag() {
        // A → {B, C} → D
        let batches = topologically_batch(vec![
            action("a", &[]),
            action("b", &["a"]),
            action("c", &["a"]),
            action("d", &["b", "c"]),
        ]).unwrap();
        assert_eq!(batches.len(), 3);
        // Batch 1: A
        assert_eq!(batches[0].len(), 1);
        // Batch 2: B and C (order may vary)
        assert_eq!(batches[1].len(), 2);
        // Batch 3: D
        assert_eq!(batches[2].len(), 1);
        assert_eq!(batches[2][0].id, "d");
    }

    #[test]
    fn test_independent_actions_single_batch() {
        // A, B, C all independent
        let batches = topologically_batch(vec![
            action("a", &[]),
            action("b", &[]),
            action("c", &[]),
        ]).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 3);
    }

    #[test]
    fn test_unknown_dependency_fails() {
        let result = topologically_batch(vec![
            action("a", &["nonexistent"]),
        ]);
        assert!(result.is_err());
    }
}
