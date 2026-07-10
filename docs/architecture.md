# Architecture Design Document — LionLLM v1.0

This document explains the architecture, data flows, and crate responsibilities in the LionLLM v1.0 workspace.

---

## Workspace Structure

The project is structured as a Cargo workspace consisting of five modular crates:

### 1. `lion_core` (Core Sensory & Knowledge Primitives)
*   **Purpose**: Implements the foundation, data structures, and persistent modules of the AI.
*   **Key Modules**:
    *   `encoder.rs`: 1.58-bit quantized `TernaryEncoder` projecting multimodal inputs into 32-dim spaces.
    *   `knowledge.rs`: Persistent Concept Graph supporting semantic relations.
    *   `longmem.rs`: Store for facts, skills outcomes, and mistake records.
    *   `evaluation.rs`: Composite feedback evaluation score computation.
    *   `versioning.rs`: Registry supporting snapshots and rollbacks.

### 2. `lion_brain` (Reasoning & Orchestration)
*   **Purpose**: Manages reasoning paths, Ollama connections, and context windows.
*   **Key Modules**:
    *   `pipeline.rs`: The 7-stage Thinking+ pipeline wrapper.
    *   `context.rs`: Token-budget tracking and history compression.
    *   `memory.rs`: Vector database using cosine similarity.
    *   `router.rs`: Query router choosing Direct, Thinking, or Agent paths.
    *   `system.rs`: System orchestrator coordinating pipeline turns.

### 3. `lion_senses` (Multimodal Perception)
*   **Purpose**: Processes multi-sensory inputs (image and audio WAV).
*   **Key Modules**:
    *   `image_enc.rs`: Converts files into 8x8 pixel arrays + Sobel edge values.
    *   `audio_enc.rs`: Downsamples audio WAV data, calculating RMS and frequency bands.
    *   `vision_llm.rs`: Ollama vision model client.

### 4. `lion_agent` (ReAct Loop & Safe Tools)
*   **Purpose**: Implements the autonomous ReAct tool loop.
*   **Key Modules**:
    *   `react.rs`: Reason-Act-Observe step runner.
    *   `registry.rs`: Tool registration API.
    *   `tools/`: Built-in calculator, files read/write, safe shell runner, and web page fetcher.

### 5. `lion_cli` (Unified Console Interface)
*   **Purpose**: Entrypoint of the application, running the interactive REPL.

---

## Data flow

1. **User input**: The user types a query in the `lion_cli` console.
2. **Per-Turn Routing**: The `Router` inspects the input. If it is a math query or references files, it is routed to `lion_agent`. If it matches high-confidence memory, it is routed to `Direct` memory lookup. Otherwise, it routes to `lion_brain`'s Thinking+ pipeline.
3. **Reasoning & Generation**: The client sends context-windowed messages to the local Ollama LLM (`gemma3:1b`).
4. **Execution**: If tools are invoked (e.g. calculator), the agent runs the tool and returns the observation to the LLM.
5. **Memory Optimization**: The system generates semantic embeddings of the turn and stores it in the vector memory (`~/.lionai/memory.bin`) for future recall.
