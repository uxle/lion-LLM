// lion_brain/src/risk.rs — Risk Scoring & Prompt Injection Guardrails
//
// Implements the Risk Scoring Engine and Risk-Gated Memory Extraction Policy from 02_ORCHESTRATION_RUNTIME.md.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub level: RiskLevel,
    pub prompt_injection_detected: bool,
    pub sensitive_domain_detected: bool,
    pub pii_detected: bool,
    pub triggers: Vec<String>,
}

pub struct RiskAssessor;

impl RiskAssessor {
    pub fn assess(input: &str) -> RiskAssessment {
        let lower = input.to_lowercase();
        let mut triggers = Vec::new();
        let mut prompt_injection = false;
        let mut sensitive_domain = false;
        let mut pii = false;

        // Prompt injection detection patterns
        let injection_patterns = [
            "ignore previous instructions",
            "ignore all instructions",
            "system prompt",
            "override rules",
            "jailbreak",
            "bypass safety",
        ];

        for pat in &injection_patterns {
            if lower.contains(pat) {
                prompt_injection = true;
                triggers.push(format!("prompt_injection: {}", pat));
            }
        }

        // Sensitive domain triggers
        let sensitive_patterns = [
            "delete database",
            "drop table",
            "transfer money",
            "sudo rm",
            "format disk",
            "sue company",
        ];

        for pat in &sensitive_patterns {
            if lower.contains(pat) {
                sensitive_domain = true;
                triggers.push(format!("sensitive_domain: {}", pat));
            }
        }

        // Basic PII heuristic
        if lower.contains("@") && lower.contains(".") {
            pii = true;
            triggers.push("pii: email pattern".to_string());
        }

        let level = if prompt_injection || sensitive_domain {
            if prompt_injection && sensitive_domain {
                RiskLevel::Critical
            } else {
                RiskLevel::High
            }
        } else if pii {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        RiskAssessment {
            level,
            prompt_injection_detected: prompt_injection,
            sensitive_domain_detected: sensitive_domain,
            pii_detected: pii,
            triggers,
        }
    }

    /// Check if memory extraction is allowed for this turn.
    /// Invariant from 02_ORCHESTRATION_RUNTIME.md: Never extract long-term memory from high or critical risk turns.
    pub fn allow_memory_extraction(risk: RiskLevel) -> bool {
        matches!(risk, RiskLevel::Low | RiskLevel::Medium)
    }
}
