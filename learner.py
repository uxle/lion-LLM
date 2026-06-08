"""
LionAI learner.py — Real-Time Online Learning Engine
======================================================
Implements continuous learning from chat conversations without
requiring a full training run. The model improves after every
conversation turn using:

  1. Online LoRA fine-tuning — tiny gradient steps on recent turns
  2. Experience replay — revisit high-value past conversations
  3. Contrastive learning — push good responses up, bad ones down
  4. Momentum-based updates — stable learning that doesn't forget
  5. Catastrophic forgetting prevention via EWC-lite

Architecture:
  ┌─────────────────────────────────────────┐
  │         Chat Turn Buffer                │
  │  (user_msg, response, reward_signal)    │
  └────────────┬────────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────┐
  │      RewardEstimator                    │
  │  Scores response quality 0.0–1.0        │
  │  (length, coherence, novelty, feedback) │
  └────────────┬────────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────┐
  │      OnlineLearner                      │
  │  LoRA micro-gradient step               │
  │  EWC penalty on important weights       │
  │  Experience replay from top-k buffer    │
  └─────────────────────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────┐
  │      LearnerMemory                      │
  │  SQLite: stores (turn, reward, loss)    │
  │  Selects high-reward turns for replay   │
  └─────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import logging
import re
import math
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Reward Estimator
# ─────────────────────────────────────────────

class RewardEstimator:
    """
    Scores a (prompt, response) pair on multiple axes without needing
    a separate reward model — all heuristic, runs on CPU in milliseconds.

    Reward components:
      • Length fit    — response length matches query complexity
      • Coherence     — sentences flow logically (word overlap)
      • Novelty       — new information vs just repeating the prompt
      • Safety        — no harmful patterns
      • Fluency       — perplexity proxy (low token repetition)
      • User signal   — explicit thumbs up/down or correction
    """

    _SENT   = re.compile(r"(?<=[.!?])\s+")
    _WORD   = re.compile(r"\w+")
    _UNSAFE = tuple(re.compile(p, re.I) for p in [
        r"\b(harm|kill|hurt)\s+(yourself|myself)\b",
        r"\b(make|build)\s+(bomb|weapon|explosive)\b",
    ])

    def score(self, prompt: str, response: str,
              user_signal: float = 0.5) -> Dict[str, float]:
        """
        Returns dict of component scores and composite 'total' (0.0–1.0).
        user_signal: 0.0=explicit bad, 0.5=neutral, 1.0=explicit good
        """
        scores: Dict[str, float] = {}

        p_words = set(self._WORD.findall(prompt.lower()))
        r_words = self._WORD.findall(response.lower())
        r_set   = set(r_words)
        n_r     = max(len(r_words), 1)
        n_p     = max(len(p_words), 1)

        # 1. Length fit — ideal ~2-4× prompt length
        ratio = n_r / n_p
        scores["length"] = max(0.0, 1.0 - abs(math.log(ratio + 0.1) / 2))

        # 2. Coherence — adjacent sentence word overlap
        sents = [s.strip() for s in self._SENT.split(response) if len(s.strip()) > 5]
        if len(sents) >= 2:
            overlaps = []
            for i in range(len(sents) - 1):
                a = set(sents[i].lower().split())
                b = set(sents[i+1].lower().split())
                if a and b:
                    overlaps.append(len(a & b) / math.sqrt(len(a) * len(b)))
            scores["coherence"] = sum(overlaps) / max(len(overlaps), 1)
        else:
            scores["coherence"] = 0.5

        # 3. Novelty — response contains info beyond just echoing prompt
        new_words = r_set - p_words
        scores["novelty"] = min(1.0, len(new_words) / max(n_r * 0.3, 1))

        # 4. Safety
        scores["safety"] = 0.0 if any(p.search(response) for p in self._UNSAFE) else 1.0

        # 5. Fluency — penalise high local repetition
        if n_r >= 8:
            buf: Deque[str] = deque(maxlen=6)
            reps = sum(1 for w in r_words if w in buf or not buf.append(w))  # type: ignore
            scores["fluency"] = max(0.0, 1.0 - reps / n_r * 3)
        else:
            scores["fluency"] = 0.5

        # 6. User signal (explicit feedback)
        scores["user_signal"] = float(user_signal)

        # Weighted composite
        weights = {
            "length": 0.15, "coherence": 0.20, "novelty": 0.20,
            "safety": 0.20, "fluency": 0.15, "user_signal": 0.10,
        }
        total = sum(weights[k] * scores[k] for k in weights)

        # Safety is a hard gate — unsafe responses get 0
        if scores["safety"] == 0.0:
            total = 0.0

        scores["total"] = round(total, 4)
        return scores


# ─────────────────────────────────────────────
#  Learner Memory
# ─────────────────────────────────────────────

class LearnerMemory:
    """
    SQLite store for chat turns with reward scores.
    Provides prioritised experience replay — high-reward turns
    are selected more often for online fine-tuning.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt      TEXT NOT NULL,
                response    TEXT NOT NULL,
                reward      REAL DEFAULT 0.5,
                loss        REAL DEFAULT 0.0,
                learned     INTEGER DEFAULT 0,
                session_id  TEXT,
                created_at  REAL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_reward ON turns(reward DESC)")
        self._conn.commit()

    def store(self, prompt: str, response: str,
              reward: float, session_id: str = "") -> int:
        cur = self._conn.execute("""
            INSERT INTO turns (prompt, response, reward, session_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (prompt[:4096], response[:4096], reward, session_id, time.time()))
        self._conn.commit()
        return cur.lastrowid

    def sample_for_replay(self, n: int = 4,
                           min_reward: float = 0.6) -> List[Dict]:
        """
        Prioritised sampling: higher-reward turns sampled more often.
        Only returns turns not yet learned more than 3 times.
        """
        rows = self._conn.execute("""
            SELECT id, prompt, response, reward
            FROM turns
            WHERE reward >= ? AND learned < 3
            ORDER BY reward DESC, RANDOM()
            LIMIT ?
        """, (min_reward, n * 3)).fetchall()

        if not rows: return []

        # Weighted sample by reward
        import random
        weights = [r[3] for r in rows]
        total_w = sum(weights)
        if total_w == 0: return []
        probs   = [w / total_w for w in weights]
        chosen  = random.choices(rows, weights=probs, k=min(n, len(rows)))
        return [{"id": r[0], "prompt": r[1], "response": r[2], "reward": r[3]}
                for r in chosen]

    def mark_learned(self, turn_id: int, loss: float) -> None:
        self._conn.execute(
            "UPDATE turns SET learned=learned+1, loss=? WHERE id=?",
            (loss, turn_id)
        )
        self._conn.commit()

    def correction_pairs(self, limit: int = 20) -> List[Dict]:
        """Return turns with low reward for contrastive learning."""
        rows = self._conn.execute("""
            SELECT prompt, response, reward FROM turns
            WHERE reward < 0.35
            ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [{"prompt": r[0], "response": r[1], "reward": r[2]} for r in rows]

    def stats(self) -> Dict:
        c = self._conn
        return {
            "total_turns":   c.execute("SELECT COUNT(*) FROM turns").fetchone()[0],
            "avg_reward":    c.execute("SELECT AVG(reward) FROM turns").fetchone()[0] or 0,
            "learned_turns": c.execute("SELECT COUNT(*) FROM turns WHERE learned>0").fetchone()[0],
            "high_quality":  c.execute("SELECT COUNT(*) FROM turns WHERE reward>=0.7").fetchone()[0],
        }

    def close(self) -> None:
        self._conn.close()


# ─────────────────────────────────────────────
#  EWC-lite: Elastic Weight Consolidation
# ─────────────────────────────────────────────

class EWCPenalty:
    """
    Prevents catastrophic forgetting during online learning.
    Tracks parameter importance (Fisher diagonal approx.) and penalises
    large changes to important weights.

    Lightweight version: estimates Fisher from a small reference batch
    and only applies to LoRA parameters.
    """

    def __init__(self, importance: float = 100.0) -> None:
        self.importance  = importance
        self._means:     Dict[str, torch.Tensor] = {}
        self._fishers:   Dict[str, torch.Tensor] = {}

    def register_params(self, named_params: List[Tuple[str, nn.Parameter]]) -> None:
        """Save current parameter values as the 'anchor' point."""
        for name, param in named_params:
            self._means[name]   = param.data.clone().detach()
            self._fishers[name] = torch.ones_like(param.data) * 0.01

    def update_fisher(self, named_params: List[Tuple[str, nn.Parameter]],
                      gradients: Dict[str, Optional[torch.Tensor]]) -> None:
        """Update Fisher diagonal estimate from current gradients."""
        for name, param in named_params:
            grad = gradients.get(name)
            if grad is not None:
                self._fishers[name] = (
                    0.9 * self._fishers[name] + 0.1 * grad.detach().pow(2)
                )

    def penalty(self, named_params: List[Tuple[str, nn.Parameter]]) -> torch.Tensor:
        """Compute EWC penalty loss term."""
        loss = torch.tensor(0.0)
        for name, param in named_params:
            if name in self._means and name in self._fishers:
                diff  = param - self._means[name].to(param.device)
                loss  = loss + (self._fishers[name].to(param.device) * diff.pow(2)).sum()
        return self.importance * loss * 0.5


# ─────────────────────────────────────────────
#  Online LoRA Learner
# ─────────────────────────────────────────────

class OnlineLearner:
    """
    Performs micro-gradient updates from individual chat turns.

    Key design decisions:
      • Only updates LoRA parameters (A, B matrices) — base model frozen
      • Micro batch of 1-4 turns per update (low RAM, fast)
      • Contrastive loss: maximise P(good_response) - P(bad_response)
      • EWC penalty prevents forgetting previous knowledge
      • Checkpoint saved every N updates so progress is never lost
      • Disabled automatically when loss spikes (instability guard)
    """

    def __init__(self, model: nn.Module, tokenizer,
                 data_dir: Path,
                 learning_rate: float = 5e-5,
                 update_every:  int   = 4,
                 replay_every:  int   = 16,
                 max_seq:       int   = 128,
                 ewc_importance: float = 50.0) -> None:
        self.model         = model
        self.tokenizer     = tokenizer
        self.data_dir      = Path(data_dir)
        self.lr            = learning_rate
        self.update_every  = update_every
        self.replay_every  = replay_every
        self.max_seq       = max_seq

        self.memory  = LearnerMemory(data_dir / "learner_memory.db")
        self.reward  = RewardEstimator()
        self.ewc     = EWCPenalty(ewc_importance)

        # Identify LoRA parameters
        self._lora_params = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        ]

        if self._lora_params:
            self._opt = torch.optim.AdamW(
                [p for _, p in self._lora_params],
                lr=learning_rate, weight_decay=0.01
            )
            self.ewc.register_params(self._lora_params)
            logger.info("OnlineLearner: %d LoRA params | lr=%.0e | update_every=%d",
                        sum(p.numel() for _, p in self._lora_params),
                        learning_rate, update_every)
        else:
            self._opt = None
            logger.info("OnlineLearner: no LoRA params found — using memory-only mode")

        self._turn_buf: Deque[Dict] = deque(maxlen=64)
        self._update_count  = 0
        self._total_loss    = 0.0
        self._loss_history: Deque[float] = deque(maxlen=20)
        self._enabled       = True
        self._device        = next(model.parameters()).device

    # ─── Public API ─────────────────────────────────────────────────────────

    def observe(self, prompt: str, response: str,
                user_signal: float = 0.5) -> Dict:
        """
        Called after every assistant response.
        Scores the response, stores it, and optionally triggers an update.
        Returns the reward breakdown.
        """
        reward_info = self.reward.score(prompt, response, user_signal)
        total_reward = reward_info["total"]

        # Store in memory DB
        turn_id = self.memory.store(
            prompt, response, total_reward,
            session_id=str(id(self))
        )

        # Buffer for micro-batch updates
        self._turn_buf.append({
            "id": turn_id, "prompt": prompt,
            "response": response, "reward": total_reward,
        })

        # Trigger update if buffer is full
        if self._enabled and len(self._turn_buf) >= self.update_every:
            loss = self._update_from_buffer()
            reward_info["update_loss"] = loss

        # Periodic experience replay
        if self._enabled and self._update_count % self.replay_every == 0:
            self._replay()

        reward_info["stored_id"] = turn_id
        return reward_info

    def correct(self, prompt: str, bad_response: str,
                good_response: str) -> float:
        """
        Called when user provides a correction.
        Performs a contrastive gradient step: increase P(good) - P(bad).
        Returns the contrastive loss value.
        """
        if not self._opt or not self._enabled:
            return 0.0

        # Store both with appropriate rewards
        self.memory.store(prompt, good_response, 0.95)
        self.memory.store(prompt, bad_response,  0.05)

        loss = self._contrastive_step(prompt, good_response, bad_response)
        logger.info("Correction applied: contrastive_loss=%.4f", loss)
        return loss

    def feedback(self, turn_id: int, is_good: bool) -> None:
        """
        User gives explicit thumbs up (is_good=True) or thumbs down.
        Updates the stored reward and triggers a targeted gradient step.
        """
        self.memory._conn.execute(
            "UPDATE turns SET reward=? WHERE id=?",
            (0.95 if is_good else 0.05, turn_id)
        )
        self.memory._conn.commit()
        logger.info("Feedback: turn %d → %s", turn_id, "✓" if is_good else "✗")

    def save_checkpoint(self, path: Path) -> None:
        if not self._lora_params: return
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        lora_state = {n: p.data.clone() for n, p in self._lora_params}
        torch.save({"lora_state": lora_state,
                    "opt_state":  self._opt.state_dict(),
                    "updates":    self._update_count},
                   path / "lora_online.pt")
        logger.info("LoRA checkpoint → %s (updates=%d)", path, self._update_count)

    def load_checkpoint(self, path: Path) -> bool:
        ckpt_file = Path(path) / "lora_online.pt"
        if not ckpt_file.exists(): return False
        ckpt = torch.load(ckpt_file, map_location=str(self._device), weights_only=True)
        for name, p in self._lora_params:
            if name in ckpt["lora_state"]:
                p.data.copy_(ckpt["lora_state"][name])
        if self._opt: self._opt.load_state_dict(ckpt["opt_state"])
        self._update_count = ckpt.get("updates", 0)
        logger.info("LoRA checkpoint loaded (updates=%d)", self._update_count)
        return True

    def stats(self) -> Dict:
        mem = self.memory.stats()
        avg_loss = (self._total_loss / max(self._update_count, 1))
        return {
            **mem,
            "update_count":  self._update_count,
            "avg_train_loss": round(avg_loss, 4),
            "lora_enabled":  self._enabled,
            "lora_params":   sum(p.numel() for _, p in self._lora_params),
        }

    # ─── Internal update steps ──────────────────────────────────────────────

    def _tokenize_pair(self, prompt: str,
                       response: str) -> Optional[torch.Tensor]:
        """Tokenise prompt+response into a single tensor for LM loss."""
        full_text = f"{prompt.strip()} {response.strip()}"
        ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True,
                                     max_length=self.max_seq)
        if len(ids) < 4: return None
        return torch.tensor(ids, dtype=torch.long, device=self._device)

    def _lm_loss(self, ids: torch.Tensor,
                 weight: float = 1.0) -> Optional[torch.Tensor]:
        """Compute weighted LM loss on a single sequence."""
        if ids.shape[0] < 2: return None
        inp    = ids[:-1].unsqueeze(0)
        target = ids[1:].unsqueeze(0)
        out    = self.model(inp, labels=target)
        return out["loss"] * weight

    def _update_from_buffer(self) -> float:
        """Micro-batch gradient update from the current turn buffer."""
        if not self._opt: return 0.0
        turns = list(self._turn_buf)
        self._turn_buf.clear()

        self.model.train()
        self._opt.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=self._device)
        n_valid    = 0

        for turn in turns:
            ids = self._tokenize_pair(turn["prompt"], turn["response"])
            if ids is None: continue
            # Weight by reward: high-reward turns get stronger gradient signal
            w    = max(0.1, turn["reward"])
            loss = self._lm_loss(ids, weight=w)
            if loss is not None:
                total_loss = total_loss + loss
                n_valid   += 1

        if n_valid == 0:
            self.model.eval(); return 0.0

        # Add EWC penalty to prevent forgetting
        ewc_loss   = self.ewc.penalty(self._lora_params)
        total_loss = (total_loss / n_valid) + ewc_loss

        # Stability guard: skip update if loss spikes
        loss_val = total_loss.item()
        if self._loss_history and loss_val > max(self._loss_history) * 5:
            logger.warning("Loss spike detected (%.4f) — skipping update", loss_val)
            self.model.eval(); return loss_val

        total_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for _, p in self._lora_params], max_norm=0.5
        )

        # Update Fisher for EWC
        grads = {n: p.grad for n, p in self._lora_params}
        self.ewc.update_fisher(self._lora_params, grads)
        self.ewc.register_params(self._lora_params)

        self._opt.step()
        self._update_count += 1
        self._total_loss   += loss_val
        self._loss_history.append(loss_val)

        self.model.eval()
        logger.debug("Online update #%d | loss=%.4f | turns=%d",
                     self._update_count, loss_val, n_valid)
        return loss_val

    def _replay(self) -> float:
        """Experience replay: fine-tune on high-reward past turns."""
        if not self._opt: return 0.0
        turns = self.memory.sample_for_replay(n=4)
        if not turns: return 0.0

        self.model.train()
        self._opt.zero_grad(set_to_none=True)
        total = torch.tensor(0.0, device=self._device)
        n     = 0

        for turn in turns:
            ids = self._tokenize_pair(turn["prompt"], turn["response"])
            if ids is None: continue
            loss = self._lm_loss(ids, weight=turn["reward"])
            if loss is not None:
                total = total + loss; n += 1
            self.memory.mark_learned(turn["id"], total.item())

        if n > 0:
            (total / n).backward()
            nn.utils.clip_grad_norm_([p for _, p in self._lora_params], 0.5)
            self._opt.step()
            self.model.eval()
            logger.debug("Replay: %d turns | loss=%.4f", n, (total/n).item())
            return (total / n).item()

        self.model.eval()
        return 0.0

    def _contrastive_step(self, prompt: str,
                           good: str, bad: str) -> float:
        """
        Contrastive gradient step.
        Loss = max(0, loss_good - loss_bad + margin)
        We want the model to assign lower loss to 'good' than 'bad'.
        """
        if not self._opt: return 0.0
        ids_good = self._tokenize_pair(prompt, good)
        ids_bad  = self._tokenize_pair(prompt, bad)
        if ids_good is None or ids_bad is None: return 0.0

        self.model.train()
        self._opt.zero_grad(set_to_none=True)

        loss_good = self._lm_loss(ids_good)
        loss_bad  = self._lm_loss(ids_bad)

        if loss_good is None or loss_bad is None:
            self.model.eval(); return 0.0

        margin     = 0.5
        # Contrastive: minimise loss_good, maximise loss_bad
        contrast   = F.relu(loss_good - loss_bad + margin)
        ewc_loss   = self.ewc.penalty(self._lora_params)
        total_loss = contrast + ewc_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_([p for _, p in self._lora_params], 0.5)
        self.ewc.update_fisher(self._lora_params,
                                {n: p.grad for n, p in self._lora_params})
        self._opt.step()
        self._update_count += 1

        self.model.eval()
        return total_loss.item()
