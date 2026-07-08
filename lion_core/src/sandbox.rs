// lion_core/src/sandbox.rs

use crate::brain::BrainMatrix;
use crate::constants::{
    BASE_MUTATION_RATE, EVOLUTION_MARGIN, FEATURE_SIZE,
    FITNESS_EVASION_REWARD, FITNESS_NEURON_COST,
    FITNESS_REPEAT_PENALTY, FITNESS_SYNAPSE_COST,
    SYNAPSE_PRUNE_THRESHOLD,
    WEIGHT_MAX, WEIGHT_MIN,
};
use crate::episode::EpisodicBuffer;
use crate::neuron::ActionLabel;
use crate::rng::BrainRng;
use crate::types::{GenIndex, Role};
use crate::workspace::best_motor_action;

// =============================================================================
// NIGHT CYCLE REPORT
// =============================================================================

/// A summary of what happened during the evolutionary night cycle.
#[derive(Debug, Clone)]
pub struct NightCycleReport {
    pub sovereign_fitness:  f64,
    pub best_child_fitness: f64,
    pub children_evaluated: usize,
    pub evolution_occurred: bool,
    pub mutation_rate:      f32,
}

impl NightCycleReport {
    pub fn fitness_gain(&self) -> f64 {
        if self.evolution_occurred {
            self.best_child_fitness - self.sovereign_fitness
        } else {
            0.0
        }
    }
}

// =============================================================================
// FITNESS EVALUATION
// =============================================================================

/// Computes the fitness of a brain + language motor against episode history.
pub fn evaluate_fitness(
    brain:          &mut BrainMatrix,
    episodes:       &EpisodicBuffer,
    language_motor: &crate::language::LanguageMotor,
) -> f64 {
    let plasticity  = brain.epigenome.plasticity;
    let mut fitness = 0.0_f64;

    for episode in episodes.as_slice() {
        brain.reset_activations();
        brain.inject_frame_no_trace(&episode.sensory_inputs);
        brain.propagate_with_hebbian(2, plasticity);

        let child_action = best_motor_action(brain);
        let taken_action = episode.action_str();
        let reward       = episode.reward_received as f64;

        if episode.was_positive() {
            if child_action == taken_action {
                fitness += reward;
            }
        } else if episode.was_negative() {
            if child_action == taken_action {
                fitness -= reward.abs() * FITNESS_REPEAT_PENALTY;
            } else {
                fitness += reward.abs() * FITNESS_EVASION_REWARD;
            }
        }

        // Language fitness: bigram-similarity to expected speech for this action.
        let (gestalt, _) = crate::workspace::gather_consciousness(
            brain, crate::workspace::WORKSPACE_TOP_K,
        );
        let action_target = crate::language::target_speech_for_action(taken_action);
        let lang_score    = language_motor.compute_language_score(&gestalt, action_target);
        fitness += lang_score * 20.0; // Strong signal — language evolution needs pressure
    }

    let synapse_count = brain.alive_synapse_count() as f64;
    let neuron_count  = brain.alive_neuron_count()  as f64;
    fitness -= (synapse_count * FITNESS_SYNAPSE_COST) + (neuron_count * FITNESS_NEURON_COST);

    fitness
}

// =============================================================================
// GRAPH MUTATION
// =============================================================================

pub fn mutate_graph(brain: &mut BrainMatrix, rng: &mut BrainRng, mut_rate: f32) {
    perturb_weights(brain, rng, mut_rate);
    prune_weak_synapses(brain);
    grow_random_synapse(brain, rng, mut_rate);
    attempt_mitosis_population(brain, rng, mut_rate);
}

fn perturb_weights(brain: &mut BrainMatrix, rng: &mut BrainRng, mut_rate: f32) {
    for syn in brain.synapses.iter_mut() {
        if !syn.alive { continue; }
        if rng.gen_bool_with_prob(mut_rate) {
            syn.weight = (syn.weight + rng.gen_mutation_delta()).clamp(WEIGHT_MIN, WEIGHT_MAX);
        }
    }
}

fn prune_weak_synapses(brain: &mut BrainMatrix) {
    let prune_targets: Vec<GenIndex> = brain
        .synapses
        .iter()
        .enumerate()
        .filter(|(_, syn)| syn.alive && syn.weight.abs() < SYNAPSE_PRUNE_THRESHOLD)
        .map(|(idx, _)| GenIndex { index: idx, generation: brain.synapse_generations[idx] })
        .collect();

    for id in prune_targets {
        brain.remove_synapse(id);
    }
}

fn grow_random_synapse(brain: &mut BrainMatrix, rng: &mut BrainRng, mut_rate: f32) {
    if !rng.gen_bool_with_prob(mut_rate) { return; }

    let ids: Vec<GenIndex> = brain.collect_alive_neuron_ids();
    if ids.len() < 2 { return; }

    let pre  = ids[rng.gen_index(ids.len())];
    let post = ids[rng.gen_index(ids.len())];
    if pre == post { return; }

    let _ = brain.insert_synapse(pre, post, rng.gen_initial_weight());
}

// =============================================================================
// MITOSIS
// =============================================================================

struct MitosisParentData {
    role:         Role,
    generation:   u32,
    base_vector:  [f32; FEATURE_SIZE],
    action_label: Option<ActionLabel>,
    trace_count:  usize,
    traces:       [crate::types::MemoryTrace; crate::constants::MAX_TRACES],
}

fn attempt_mitosis_population(brain: &mut BrainMatrix, rng: &mut BrainRng, mut_rate: f32) {
    let candidates: Vec<GenIndex> = brain
        .alive_neurons()
        .filter(|n| n.is_overloaded())
        .map(|n| n.id)
        .collect();

    for parent_id in candidates {
        if !brain.is_valid_neuron(parent_id) { continue; }
        if !rng.gen_bool_with_prob(mut_rate) { continue; }
        perform_mitosis(brain, parent_id, rng);
    }
}

fn perform_mitosis(brain: &mut BrainMatrix, parent_id: GenIndex, rng: &mut BrainRng) {
    let parent_data = {
        let p = &brain.neurons[parent_id.index];
        MitosisParentData {
            role:         p.role,
            generation:   p.generation,
            base_vector:  p.base_vector,
            action_label: p.action_label,
            trace_count:  p.trace_count,
            traces:       p.traces,
        }
    };

    let outgoing: Vec<(GenIndex, f32)> = brain
        .synapses.iter()
        .filter(|s| s.alive && s.pre_id == parent_id)
        .map(|s| (s.post_id, s.weight))
        .collect();

    let incoming: Vec<(GenIndex, f32)> = brain
        .synapses.iter()
        .filter(|s| s.alive && s.post_id == parent_id)
        .map(|s| (s.pre_id, s.weight))
        .collect();

    let mut child_base = parent_data.base_vector;
    for x in child_base.iter_mut() { *x += rng.gen_mitosis_jitter(); }

    let child_id = match brain.insert_neuron(parent_data.role, child_base) {
        Some(id) => id,
        None     => return,
    };

    {
        let child = &mut brain.neurons[child_id.index];
        child.generation   = parent_data.generation + 1;
        child.action_label = parent_data.action_label;

        let half              = parent_data.trace_count / 2;
        let child_trace_count = parent_data.trace_count - half;
        child.trace_count     = child_trace_count;
        child.traces[..child_trace_count]
            .copy_from_slice(&parent_data.traces[half..parent_data.trace_count]);
    }

    brain.neurons[parent_id.index].trace_count = parent_data.trace_count / 2;

    for (post_id, weight) in outgoing {
        if brain.is_valid_neuron(post_id) {
            let jittered = (weight + rng.gen_synapse_mitosis_jitter()).clamp(WEIGHT_MIN, WEIGHT_MAX);
            let _ = brain.insert_synapse(child_id, post_id, jittered);
        }
    }

    for (pre_id, weight) in incoming {
        if brain.is_valid_neuron(pre_id) && pre_id != child_id {
            let jittered = (weight + rng.gen_synapse_mitosis_jitter()).clamp(WEIGHT_MIN, WEIGHT_MAX);
            let _ = brain.insert_synapse(pre_id, child_id, jittered);
        }
    }
}

// =============================================================================
// NIGHT CYCLE
// =============================================================================

/// Derives a unique child seed from a base seed and index.
#[inline]
pub fn child_seed(base_seed: u64, child_idx: usize) -> u64 {
    base_seed
        .wrapping_add(child_idx as u64 + 1)
        .wrapping_mul(crate::constants::LCG_MULTIPLIER)
}

/// Sequential night cycle (kept for API compatibility).
pub fn run_night_cycle(
    sovereign_brain:    &BrainMatrix,
    sovereign_language: &crate::language::LanguageMotor,
    episodes:           &EpisodicBuffer,
    rng:                &mut BrainRng,
    population_size:    usize,
) -> (Option<BrainMatrix>, Option<crate::language::LanguageMotor>, NightCycleReport) {
    if episodes.is_empty() {
        return (None, None, NightCycleReport {
            sovereign_fitness: 0.0, best_child_fitness: 0.0,
            children_evaluated: 0, evolution_occurred: false, mutation_rate: 0.0,
        });
    }

    let mut eval_brain    = sovereign_brain.clone();
    let sovereign_fitness = evaluate_fitness(&mut eval_brain, episodes, sovereign_language);
    let mutation_rate     = sovereign_brain.epigenome.effective_mutation_rate(BASE_MUTATION_RATE);

    let mut best_fitness = sovereign_fitness;
    let mut best_brain:  Option<BrainMatrix>                   = None;
    let mut best_lang:   Option<crate::language::LanguageMotor> = None;

    for _ in 0..population_size {
        let mut child_brain = sovereign_brain.clone();
        let mut child_lang  = sovereign_language.clone();

        mutate_graph(&mut child_brain, rng, mutation_rate);
        child_lang.mutate(rng, mutation_rate);

        let pd = rng.gen_plasticity_delta();
        let ed = rng.gen_exploration_delta();
        child_brain.epigenome.mutate(pd, ed);

        let child_fitness = evaluate_fitness(&mut child_brain, episodes, &child_lang);

        if child_fitness > best_fitness + EVOLUTION_MARGIN {
            best_fitness = child_fitness;
            best_brain   = Some(child_brain);
            best_lang    = Some(child_lang);
        }
    }

    let evolution_occurred = best_brain.is_some();
    (best_brain, best_lang, NightCycleReport {
        sovereign_fitness,
        best_child_fitness: best_fitness,
        children_evaluated: population_size,
        evolution_occurred,
        mutation_rate,
    })
}

// =============================================================================
// PARALLEL NIGHT CYCLE (Phase 9)
// =============================================================================

use rayon::prelude::*;
use std::cmp::Ordering;

/// Parallel night cycle — evaluates all children concurrently across all CPU cores.
pub fn run_night_cycle_parallel(
    sovereign_brain:    &BrainMatrix,
    sovereign_language: &crate::language::LanguageMotor,
    episodes:           &EpisodicBuffer,
    base_seed:          u64,
    population_size:    usize,
) -> (Option<BrainMatrix>, Option<crate::language::LanguageMotor>, NightCycleReport) {
    if episodes.is_empty() {
        return (None, None, NightCycleReport {
            sovereign_fitness: 0.0, best_child_fitness: 0.0,
            children_evaluated: 0, evolution_occurred: false, mutation_rate: 0.0,
        });
    }

    let sovereign_fitness = {
        let mut eval = sovereign_brain.clone();
        evaluate_fitness(&mut eval, episodes, sovereign_language)
    };

    let mutation_rate = sovereign_brain.epigenome.effective_mutation_rate(BASE_MUTATION_RATE);

    let best_result: Option<(BrainMatrix, crate::language::LanguageMotor, f64)> =
        (0..population_size)
            .into_par_iter()
            .map(|i| {
                let seed       = child_seed(base_seed, i);
                let mut rng    = BrainRng::from_seed(seed);
                let mut child  = sovereign_brain.clone();
                let mut c_lang = sovereign_language.clone();

                mutate_graph(&mut child, &mut rng, mutation_rate);
                c_lang.mutate(&mut rng, mutation_rate);

                let pd = rng.gen_plasticity_delta();
                let ed = rng.gen_exploration_delta();
                child.epigenome.mutate(pd, ed);

                let fitness = evaluate_fitness(&mut child, episodes, &c_lang);
                (child, c_lang, fitness)
            })
            .max_by(|(_, _, fa), (_, _, fb)| {
                fa.partial_cmp(fb).unwrap_or(Ordering::Equal)
            });

    let (best_child_fitness, winner_brain, winner_lang) = match best_result {
        Some((brain, lang, f)) if f > sovereign_fitness + EVOLUTION_MARGIN =>
            (f, Some(brain), Some(lang)),
        Some((_, _, f)) => (f, None, None),
        None            => (sovereign_fitness, None, None),
    };

    let evolution_occurred = winner_brain.is_some();
    (winner_brain, winner_lang, NightCycleReport {
        sovereign_fitness,
        best_child_fitness,
        children_evaluated: population_size,
        evolution_occurred,
        mutation_rate,
    })
}
