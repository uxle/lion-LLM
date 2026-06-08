"""
LionAI reasoner.py — Chain-of-Thought Reasoning & Self-Correction
===================================================================
Gives LionAI human-like reasoning capabilities:

  1. Chain-of-Thought (CoT) — breaks complex questions into steps
  2. Self-Verification — checks its own answer for consistency
  3. Intent Classification — understands WHAT the user actually wants
  4. Fact-checking — cross-references knowledge base before responding
  5. Confidence scoring — knows when it doesn't know something
  6. Response planning — outlines before answering complex questions
  7. Entity extraction — tracks people, places, concepts across turns
  8. Contradiction detection — catches conflicting statements in context

Architecture:
  User query
       │
       ▼
  IntentClassifier ──► intent type + confidence
       │
       ▼
  ChainOfThought ──► scratchpad reasoning steps
       │
       ▼
  ResponsePlanner ──► structured answer plan
       │
       ▼
  LionLLM (generate)
       │
       ▼
  SelfVerifier ──► consistency check + confidence
       │
       ▼
  Final Response
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Intent Types
# ─────────────────────────────────────────────

INTENT_TYPES = {
    "question_factual":   "Asking for a fact or definition",
    "question_opinion":   "Asking for an opinion or recommendation",
    "question_how":       "Asking how to do something",
    "question_why":       "Asking for an explanation or reason",
    "task_code":          "Requesting code or debugging help",
    "task_write":         "Requesting written content",
    "task_analyse":       "Requesting analysis of provided content",
    "task_math":          "Requesting mathematical calculation",
    "conversation":       "Casual conversation or greeting",
    "correction":         "User is correcting the AI",
    "feedback_positive":  "User is expressing satisfaction",
    "feedback_negative":  "User is expressing dissatisfaction",
    "memory_store":       "User wants AI to remember something",
    "memory_query":       "User is asking about something previously said",
    "clarification":      "User is clarifying a previous statement",
    "unknown":            "Intent unclear",
}


# ─────────────────────────────────────────────
#  Intent Classifier (rule-based, zero latency)
# ─────────────────────────────────────────────

class IntentClassifier:
    """
    Classifies user intent using pattern matching + keyword analysis.
    Runs in <1ms with no model inference needed.
    """

    _PATTERNS: List[Tuple[str, List[re.Pattern]]] = [
        ("question_how",  [re.compile(r"\bhow\s+(do|can|to|does|should|would)\b", re.I),
                           re.compile(r"\bsteps?\s+to\b", re.I)]),
        ("question_why",  [re.compile(r"\bwhy\s+(is|are|does|do|did|would)\b", re.I),
                           re.compile(r"\bwhat.{1,20}reason\b", re.I)]),
        ("question_factual", [re.compile(r"\bwhat\s+(is|are|was|were)\b", re.I),
                              re.compile(r"\bwho\s+(is|was|are)\b", re.I),
                              re.compile(r"\bwhere\s+(is|are|was)\b", re.I),
                              re.compile(r"\bwhen\s+(did|was|is)\b", re.I),
                              re.compile(r"\bdefine\b|\bexplain\b|\bdescribe\b", re.I)]),
        ("task_code",     [re.compile(r"\b(write|create|fix|debug|code|function|class|script|program)\b", re.I),
                           re.compile(r"\b(python|javascript|java|c\+\+|sql|html|css)\b", re.I)]),
        ("task_math",     [re.compile(r"\b(calculate|compute|solve|math|equation|integral|derivative)\b", re.I),
                           re.compile(r"\d+\s*[\+\-\*\/\^]\s*\d+")]),
        ("task_write",    [re.compile(r"\b(write|draft|compose|create)\s+(a|an|the|some)?\s*(email|letter|essay|poem|story|paragraph|blog|summary)\b", re.I)]),
        ("task_analyse",  [re.compile(r"\b(analyse|analyze|review|evaluate|assess|summarize|summarise)\b", re.I)]),
        ("correction",    [re.compile(r"\b(no,?\s+that'?s?\s+wrong|incorrect|not right|you'?re?\s+wrong|actually)\b", re.I),
                           re.compile(r"\b(correct(ion)?|fix|should\s+be)\b", re.I)]),
        ("feedback_positive", [re.compile(r"\b(thank(s|you)|great|perfect|excellent|good\s+(job|answer|response)|well\s+done)\b", re.I)]),
        ("feedback_negative", [re.compile(r"\b(that.{0,20}(wrong|bad|terrible|awful|useless)|not\s+helpful|doesn.t\s+help)\b", re.I)]),
        ("memory_store",  [re.compile(r"\b(remember|don.t\s+forget|keep\s+in\s+mind|note\s+that)\b", re.I)]),
        ("memory_query",  [re.compile(r"\b(what\s+did\s+(i|you)\s+say|earlier\s+(i\s+said|you\s+said)|do\s+you\s+remember)\b", re.I)]),
        ("conversation",  [re.compile(r"^(hi|hello|hey|good\s+(morning|evening|afternoon|night)|how\s+are\s+you)\b", re.I)]),
        ("question_opinion", [re.compile(r"\b(what\s+do\s+you\s+think|in\s+your\s+opinion|which\s+(is|would)\s+(better|best))\b", re.I),
                              re.compile(r"\bdo\s+you\s+prefer\b", re.I)]),
    ]

    def classify(self, text: str) -> Tuple[str, float]:
        """Returns (intent_type, confidence 0-1)."""
        text = text.strip()
        scores: Dict[str, int] = {}

        for intent, patterns in self._PATTERNS:
            for pat in patterns:
                if pat.search(text):
                    scores[intent] = scores.get(intent, 0) + 1

        if not scores:
            return "unknown", 0.4

        best = max(scores, key=lambda k: scores[k])
        confidence = min(1.0, 0.5 + scores[best] * 0.2)
        return best, round(confidence, 2)

    def needs_reasoning(self, intent: str) -> bool:
        """Returns True if this intent warrants chain-of-thought reasoning."""
        return intent in ("question_why", "question_how", "task_math",
                          "task_analyse", "task_code", "question_factual")

    def is_simple(self, intent: str, text: str) -> bool:
        """Returns True if query can be answered directly without planning."""
        if intent in ("conversation", "feedback_positive", "memory_store"):
            return True
        return len(text.split()) < 8


# ─────────────────────────────────────────────
#  Entity Extractor
# ─────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts named entities and key concepts from text.
    Rule-based — no NLP library needed.
    """

    _PERSON  = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
    _NUMBER  = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand|kg|km|miles?|hours?|minutes?|seconds?|days?|years?|months?))?\b", re.I)
    _TECH    = re.compile(r"\b(Python|JavaScript|Java|C\+\+|SQL|HTML|CSS|API|GPU|CPU|RAM|AI|ML|LLM|NLP|neural\s+network|transformer|model)\b", re.I)
    _QUOTED  = re.compile(r'"([^"]{2,80})"')
    _CONCEPT = re.compile(r"\b(learning|training|inference|attention|gradient|loss|epoch|batch|tokenizer|embedding)\b", re.I)

    def extract(self, text: str) -> Dict[str, List[str]]:
        return {
            "persons":   self._PERSON.findall(text),
            "numbers":   self._NUMBER.findall(text),
            "tech":      list(set(m.lower() for m in self._TECH.findall(text))),
            "quoted":    self._QUOTED.findall(text),
            "concepts":  list(set(m.lower() for m in self._CONCEPT.findall(text))),
        }


# ─────────────────────────────────────────────
#  Chain-of-Thought Engine
# ─────────────────────────────────────────────

@dataclass
class ThoughtStep:
    step:      int
    type:      str    # "observe" | "reason" | "check" | "conclude"
    content:   str
    confidence: float = 0.8

@dataclass
class ReasoningTrace:
    query:        str
    intent:       str
    entities:     Dict[str, List[str]] = field(default_factory=dict)
    steps:        List[ThoughtStep]    = field(default_factory=list)
    plan:         List[str]            = field(default_factory=list)
    scratchpad:   str                  = ""
    confidence:   float                = 0.7
    elapsed_ms:   float                = 0.0

    def as_context(self) -> str:
        """Format the reasoning trace for injection into the model prompt."""
        if not self.steps: return ""
        lines = ["[Reasoning]"]
        for s in self.steps:
            lines.append(f"  {s.type.upper()}: {s.content}")
        if self.plan:
            lines.append("[Plan]")
            lines.extend(f"  {i+1}. {p}" for i, p in enumerate(self.plan))
        return "\n".join(lines)


class ChainOfThought:
    """
    Generates a structured reasoning trace for complex queries.
    The trace is injected into the prompt so the model 'thinks before speaking'.
    """

    def __init__(self, knowledge_engine=None) -> None:
        self.kb       = knowledge_engine
        self.intent   = IntentClassifier()
        self.entities = EntityExtractor()

    def think(self, query: str, context: str = "",
               memory_context: str = "") -> ReasoningTrace:
        """
        Build a reasoning trace for the given query.
        Returns a ReasoningTrace that can be injected into the prompt.
        """
        t0 = time.perf_counter()
        intent_type, intent_conf = self.intent.classify(query)
        entities = self.entities.extract(query)

        trace = ReasoningTrace(
            query   = query,
            intent  = intent_type,
            entities= entities,
        )

        # Step 1: Observe the query
        trace.steps.append(ThoughtStep(
            step=1, type="observe",
            content=f"Query type: {intent_type} (confidence={intent_conf:.0%}). "
                    f"Key entities: {', '.join(sum(entities.values(), [])) or 'none'}.",
            confidence=intent_conf,
        ))

        # Step 2: Check knowledge base if available
        if self.kb and self.intent.needs_reasoning(intent_type):
            kb_results = self.kb.retrieve(query, top_k=2)
            if kb_results:
                kb_summary = kb_results[0]["text"][:150] + "…"
                trace.steps.append(ThoughtStep(
                    step=2, type="check",
                    content=f"Knowledge base relevant context: {kb_summary}",
                    confidence=0.8,
                ))

        # Step 3: Reasoning based on intent
        if intent_type == "task_math":
            numbers = entities.get("numbers", [])
            trace.steps.append(ThoughtStep(
                step=3, type="reason",
                content=f"Mathematical query. Numbers found: {numbers}. "
                        f"Will compute step by step showing working.",
            ))
            trace.plan = [
                "Identify the mathematical operation required",
                "Show each calculation step clearly",
                "State the final answer explicitly",
                "Verify by checking the result makes sense",
            ]

        elif intent_type == "task_code":
            tech = entities.get("tech", [])
            trace.steps.append(ThoughtStep(
                step=3, type="reason",
                content=f"Code task. Technologies: {tech or ['general']}. "
                        f"Will write clean, commented, working code.",
            ))
            trace.plan = [
                "Understand the exact requirement",
                "Choose the correct approach",
                "Write clean, commented code",
                "Add usage example",
            ]

        elif intent_type == "question_why":
            trace.steps.append(ThoughtStep(
                step=3, type="reason",
                content="Explanation query. Will provide clear cause-and-effect reasoning.",
            ))
            trace.plan = [
                "State the direct answer first",
                "Explain the underlying reason",
                "Give a concrete example",
            ]

        elif intent_type == "task_analyse":
            trace.steps.append(ThoughtStep(
                step=3, type="reason",
                content="Analysis query. Will examine systematically.",
            ))
            trace.plan = [
                "Identify key elements to analyse",
                "Examine each element",
                "Draw conclusions",
                "Summarise findings",
            ]

        elif intent_type == "correction":
            trace.steps.append(ThoughtStep(
                step=3, type="reason",
                content="User is correcting me. Will acknowledge, understand the correction, and update.",
                confidence=0.9,
            ))
            trace.plan = [
                "Acknowledge the correction",
                "State what was wrong",
                "Provide the correct information",
            ]

        else:
            trace.steps.append(ThoughtStep(
                step=3, type="reason",
                content=f"Standard {intent_type} query. Will answer clearly and directly.",
            ))
            trace.plan = ["Answer clearly", "Be concise and helpful"]

        # Step 4: Memory context check
        if memory_context:
            trace.steps.append(ThoughtStep(
                step=4, type="check",
                content=f"Relevant memory found — will incorporate into response.",
            ))

        # Confidence estimate
        trace.confidence = min(0.95, intent_conf * 0.7 + 0.3)
        trace.elapsed_ms = (time.perf_counter() - t0) * 1000
        trace.scratchpad = trace.as_context()

        logger.debug("CoT: intent=%s conf=%.0f%% steps=%d time=%.1fms",
                     intent_type, intent_conf * 100, len(trace.steps), trace.elapsed_ms)
        return trace


# ─────────────────────────────────────────────
#  Self-Verifier
# ─────────────────────────────────────────────

class SelfVerifier:
    """
    Checks the model's response for consistency and quality issues
    AFTER generation, before showing to the user.
    Can trigger regeneration if quality is too low.
    """

    _CONTRADICTION_PAIRS = [
        (re.compile(r"\bis\b", re.I), re.compile(r"\bis\s+not\b|\bisn.t\b", re.I)),
        (re.compile(r"\bcan\b", re.I), re.compile(r"\bcannot\b|\bcan.t\b", re.I)),
        (re.compile(r"\bwill\b", re.I), re.compile(r"\bwill\s+not\b|\bwon.t\b", re.I)),
        (re.compile(r"\balways\b", re.I), re.compile(r"\bnever\b", re.I)),
    ]

    _UNCERTAINTY_MARKERS = re.compile(
        r"\b(i don.t know|i.m not sure|unclear|uncertain|may|might|possibly|perhaps|"
        r"could be|not certain|hard to say)\b", re.I
    )

    def verify(self, query: str, response: str,
               trace: Optional[ReasoningTrace] = None) -> Dict:
        """
        Analyse response quality.
        Returns dict with issues list and quality score.
        """
        issues: List[str] = []
        scores: Dict[str, float] = {}

        words  = response.split()
        n      = max(len(words), 1)

        # 1. Completeness — response addresses the query
        q_words   = set(query.lower().split())
        r_lower   = response.lower()
        addressed = sum(1 for w in q_words if w in r_lower and len(w) > 3)
        scores["completeness"] = min(1.0, addressed / max(len(q_words) * 0.3, 1))
        if scores["completeness"] < 0.2:
            issues.append("Response may not address the query")

        # 2. Coherence — no self-contradictions
        coherence = 1.0
        for pos_pat, neg_pat in self._CONTRADICTION_PAIRS:
            if pos_pat.search(response) and neg_pat.search(response):
                coherence -= 0.2
                issues.append("Possible self-contradiction detected")
        scores["coherence"] = max(0.0, coherence)

        # 3. Uncertainty detection — model doesn't know something
        uncertainty_hits = len(self._UNCERTAINTY_MARKERS.findall(response))
        scores["certainty"] = max(0.0, 1.0 - uncertainty_hits * 0.15)

        # 4. Length appropriateness
        if n < 3:
            issues.append("Response too short")
            scores["length"] = 0.3
        elif n > 500:
            issues.append("Response may be too verbose")
            scores["length"] = 0.7
        else:
            scores["length"] = 1.0

        # 5. Repetition check
        bigrams: Dict[tuple, int] = {}
        for i in range(len(words) - 1):
            bg = (words[i].lower(), words[i+1].lower())
            bigrams[bg] = bigrams.get(bg, 0) + 1
        max_repeat = max(bigrams.values()) if bigrams else 1
        if max_repeat > 4:
            issues.append("High repetition detected")
        scores["fluency"] = max(0.0, 1.0 - max_repeat * 0.1)

        # Composite quality score
        weights = {"completeness": 0.3, "coherence": 0.25,
                   "certainty": 0.2, "length": 0.15, "fluency": 0.1}
        quality = sum(weights[k] * scores[k] for k in weights)

        return {
            "quality":  round(quality, 3),
            "scores":   scores,
            "issues":   issues,
            "verified": quality >= 0.5,
            "uncertainty_markers": uncertainty_hits,
        }

    def needs_regeneration(self, verify_result: Dict,
                            threshold: float = 0.3) -> bool:
        return verify_result["quality"] < threshold


# ─────────────────────────────────────────────
#  Confidence Estimator
# ─────────────────────────────────────────────

class ConfidenceEstimator:
    """
    Estimates how confident LionAI should be in its response.
    High confidence = answer directly. Low confidence = hedge appropriately.
    """

    _KNOWN_DOMAINS = frozenset([
        "python", "mathematics", "programming", "science", "history",
        "geography", "grammar", "spelling", "definition", "explanation",
        "how to", "tutorial", "code", "algorithm", "data structure",
    ])

    _UNCERTAIN_DOMAINS = frozenset([
        "future", "prediction", "stock", "price", "weather", "politics",
        "sports score", "news", "current events", "today", "latest",
        "recent", "now", "2024", "2025", "real-time",
    ])

    def estimate(self, query: str, intent: str,
                 verify_result: Optional[Dict] = None) -> float:
        """Returns confidence 0.0–1.0."""
        q_lower = query.lower()
        base    = 0.7

        # Boost for known domains
        if any(d in q_lower for d in self._KNOWN_DOMAINS): base += 0.1
        # Reduce for uncertain domains
        if any(d in q_lower for d in self._UNCERTAIN_DOMAINS): base -= 0.3
        # Intent adjustments
        if intent == "question_factual":      base += 0.05
        elif intent == "conversation":         base += 0.1
        elif intent == "question_opinion":     base -= 0.1
        elif intent == "task_math":            base += 0.1

        # Verification adjustment
        if verify_result:
            base = base * 0.5 + verify_result.get("quality", 0.7) * 0.5

        return max(0.1, min(0.98, base))

    def confidence_prefix(self, confidence: float) -> str:
        """Generate appropriate hedging prefix based on confidence."""
        if confidence >= 0.9: return ""
        if confidence >= 0.75: return ""
        if confidence >= 0.55: return "Based on my knowledge, "
        if confidence >= 0.4:  return "I'm not entirely certain, but "
        return "I'm not sure about this, but to the best of my knowledge: "


# ─────────────────────────────────────────────
#  Main Reasoning Pipeline
# ─────────────────────────────────────────────

class ReasoningPipeline:
    """
    Orchestrates the full reasoning pipeline:
    Intent → CoT → [Generate] → Verify → Confidence

    Used by chatbot.py to wrap every model call.
    """

    def __init__(self, knowledge_engine=None) -> None:
        self.cot        = ChainOfThought(knowledge_engine)
        self.verifier   = SelfVerifier()
        self.confidence = ConfidenceEstimator()
        self.intent_clf = IntentClassifier()

    def prepare(self, query: str, memory_context: str = "",
                rag_context: str = "") -> Tuple[ReasoningTrace, str]:
        """
        Prepare for generation:
          1. Classify intent
          2. Build CoT trace
          3. Return (trace, augmented_prompt_prefix)
        """
        trace = self.cot.think(query, memory_context=memory_context)

        # Build augmented context to inject before model sees the query
        augmented_parts: List[str] = []
        if rag_context:
            augmented_parts.append(rag_context)
        if memory_context:
            augmented_parts.append(memory_context)
        if trace.scratchpad and len(query.split()) > 6:
            # Only inject CoT for non-trivial queries
            augmented_parts.append(trace.scratchpad)

        return trace, "\n\n".join(augmented_parts)

    def evaluate(self, query: str, response: str,
                 trace: ReasoningTrace) -> Tuple[Dict, float, str]:
        """
        Evaluate generated response:
          1. Verify quality
          2. Estimate confidence
          3. Return (verify_result, confidence, final_response)
        """
        verify_result = self.verifier.verify(query, response, trace)
        confidence    = self.confidence.estimate(
            query, trace.intent, verify_result
        )
        prefix        = self.confidence.confidence_prefix(confidence)
        final         = (prefix + response) if prefix else response

        return verify_result, confidence, final

    def should_use_cot(self, query: str) -> bool:
        """Quick check: is CoT worth the overhead for this query?"""
        intent, _ = self.intent_clf.classify(query)
        return (self.intent_clf.needs_reasoning(intent)
                and not self.intent_clf.is_simple(intent, query))
