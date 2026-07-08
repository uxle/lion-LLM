// lion_core/src/brain.rs

use crate::constants::{
    BASE_MUTATION_RATE, FEATURE_SIZE, MAX_NEURONS, MAX_SYNAPSES,
};
use crate::epigenome::Epigenome;
use crate::neuron::{ActionLabel, Neuron};
use crate::synapse::Synapse;
use crate::types::{GenIndex, Role};
use serde::{Deserialize, Serialize};

// =============================================================================
// BRAIN MATRIX — THE ARENA
// =============================================================================

/// The unified memory arena for the entire LionAI cognitive graph.
///
/// All neurons and synapses live in contiguous `Vec` memory.
/// No pointers. No heap allocations inside nodes.
/// References use `GenIndex` (index + generation) for stale-detection.
///
/// Translates the combined Python:
///   class NeuralGraph:
///       neurons: Dict[str, NeuronNode]
///       synapses: Dict[str, Dict[str, Synapse]]
///
/// Into a cache-friendly, borrow-checker-safe Rust structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainMatrix {
    // -------------------------------------------------------------------------
    // MEMORY ARENAS
    // -------------------------------------------------------------------------

    /// Contiguous block of all neuron slots (alive + dead).
    /// Live neurons: `alive == true`.
    /// Dead neurons: `alive == false`, slot is in `free_neuron_slots`.
    pub neurons: Vec<Neuron>,

    /// Contiguous block of all synapse slots (alive + dead).
    pub synapses: Vec<Synapse>,

    // -------------------------------------------------------------------------
    // GENERATION COUNTERS
    // -------------------------------------------------------------------------

    /// One generation counter per neuron slot.
    /// Incremented every time a slot is reused.
    /// A `GenIndex` is valid iff `neuron_generations[index] == gen_index.generation`.
    pub neuron_generations: Vec<u32>,

    /// One generation counter per synapse slot.
    pub synapse_generations: Vec<u32>,

    // -------------------------------------------------------------------------
    // FREE LISTS (O(1) ALLOCATION)
    // -------------------------------------------------------------------------

    /// Stack of dead neuron slot indices available for immediate reuse.
    /// Push on death. Pop on birth. Never reallocates.
    pub free_neuron_slots: Vec<usize>,

    /// Stack of dead synapse slot indices available for immediate reuse.
    pub free_synapse_slots: Vec<usize>,

    // -------------------------------------------------------------------------
    // EPIGENETIC STATE
    // -------------------------------------------------------------------------

    /// The dynamic phenotype governing plasticity, exploration, and stress.
    pub epigenome: Epigenome,

    // -------------------------------------------------------------------------
    // DNA CONSTANTS
    // -------------------------------------------------------------------------

    pub base_mutation_rate: f32,
    pub hebbian_rate:       f32,
    pub trace_decay_rate:   f32,

    // -------------------------------------------------------------------------
    // DIAGNOSTICS
    // -------------------------------------------------------------------------

    /// Counts how many NaN/Inf values the immune system has corrected.
    pub immune_interventions: u32,

    /// Which evolutionary generation this brain belongs to.
    pub generation: u32,
}

impl BrainMatrix {
    // =========================================================================
    // CONSTRUCTION
    // =========================================================================

    /// Creates a new BrainMatrix with pre-allocated arenas of fixed capacity.
    ///
    /// ALL slots are pre-created as dead tombstones.
    /// ALL slots are pushed into the free lists immediately.
    ///
    /// This means:
    ///   - Zero reallocation during the lifetime of the brain.
    ///   - insert_neuron() and insert_synapse() are O(1) Vec::pop() calls.
    ///   - The entire brain can be cloned in microseconds (Vec::clone).
    pub fn new() -> Self {
        // Pre-fill neuron arena with dead tombstones.
        let dead_neuron_id = GenIndex::new(0, 0);
        let dead_base = [0.0_f32; FEATURE_SIZE];
        let neurons: Vec<Neuron> = (0..MAX_NEURONS)
            .map(|i| {
                let mut n = Neuron::new(
                    GenIndex::new(i, 0),
                    Role::Vision,
                    dead_base,
                );
                n.alive = false;
                n
            })
            .collect();

        // Pre-fill synapse arena with dead tombstones.
        let synapses: Vec<Synapse> = (0..MAX_SYNAPSES)
            .map(|_| Synapse {
                pre_id:  dead_neuron_id,
                post_id: dead_neuron_id,
                weight:  0.0,
                alive:   false,
            })
            .collect();

        // Generation counters — all start at 0.
        let neuron_generations  = vec![0u32; MAX_NEURONS];
        let synapse_generations = vec![0u32; MAX_SYNAPSES];

        // Free lists — every slot is immediately available.
        // Reversed so that index 0 is popped first (bottom of stack = index MAX-1).
        let free_neuron_slots:  Vec<usize> = (0..MAX_NEURONS).rev().collect();
        let free_synapse_slots: Vec<usize> = (0..MAX_SYNAPSES).rev().collect();

        Self {
            neurons,
            synapses,
            neuron_generations,
            synapse_generations,
            free_neuron_slots,
            free_synapse_slots,
            epigenome:          Epigenome::default(),
            base_mutation_rate: BASE_MUTATION_RATE,
            hebbian_rate:       0.05,
            trace_decay_rate:   0.95,
            immune_interventions: 0,
            generation:         1,
        }
    }

    // =========================================================================
    // NEURON ALLOCATION
    // =========================================================================

    /// Allocates a new neuron slot and returns its `GenIndex`.
    ///
    /// Steps:
    ///   1. Pop a free slot index from the free list.
    ///   2. Bump the generation counter for that slot.
    ///   3. Write the new Neuron into the slot.
    ///   4. Return the GenIndex (slot + new generation).
    ///
    /// Returns `None` if the neuron arena is completely full.
    ///
    /// O(1) — never allocates heap memory.
    pub fn insert_neuron(
        &mut self,
        role:        Role,
        base_vector: [f32; FEATURE_SIZE],
    ) -> Option<GenIndex> {
        let slot = self.free_neuron_slots.pop()?;

        // Bump generation — invalidates all stale GenIndex values
        // that previously pointed at this slot.
        self.neuron_generations[slot] += 1;
        let gen = self.neuron_generations[slot];

        let id = GenIndex::new(slot, gen);
        self.neurons[slot] = Neuron::new(id, role, base_vector);

        Some(id)
    }

    /// Inserts a MOTOR neuron with an action label.
    pub fn insert_motor_neuron(
        &mut self,
        base_vector:  [f32; FEATURE_SIZE],
        action_label: &str,
    ) -> Option<GenIndex> {
        let id = self.insert_neuron(Role::Motor, base_vector)?;
        self.neurons[id.index].action_label = Some(ActionLabel::new(action_label));
        Some(id)
    }

    /// Removes a neuron from the arena and returns its slot to the free list.
    ///
    /// Returns `true` if the removal succeeded.
    /// Returns `false` if the `GenIndex` is stale (already dead or wrong generation).
    ///
    /// This does NOT automatically remove the neuron's synapses.
    /// The caller is responsible for cleaning up connections.
    ///
    /// O(1).
    pub fn remove_neuron(&mut self, id: GenIndex) -> bool {
        if !self.is_valid_neuron(id) {
            return false;
        }
        self.neurons[id.index].alive = false;
        self.free_neuron_slots.push(id.index);
        true
    }

    // =========================================================================
    // SYNAPSE ALLOCATION
    // =========================================================================

    /// Allocates a new synapse between two neurons and returns its `GenIndex`.
    ///
    /// Validates that BOTH endpoint neurons are alive before allocating.
    /// Returns `None` if either endpoint is dead/invalid or the arena is full.
    ///
    /// O(1).
    pub fn insert_synapse(
        &mut self,
        pre_id:  GenIndex,
        post_id: GenIndex,
        weight:  f32,
    ) -> Option<GenIndex> {
        // Both endpoints must be alive.
        if !self.is_valid_neuron(pre_id) || !self.is_valid_neuron(post_id) {
            return None;
        }
        // Self-loops are not allowed.
        if pre_id == post_id {
            return None;
        }

        let slot = self.free_synapse_slots.pop()?;

        self.synapse_generations[slot] += 1;
        let gen = self.synapse_generations[slot];

        self.synapses[slot] = Synapse::new(pre_id, post_id, weight);

        Some(GenIndex::new(slot, gen))
    }

    /// Removes a synapse from the arena.
    ///
    /// Returns `true` if the removal succeeded.
    /// Returns `false` if the `GenIndex` is stale.
    ///
    /// O(1).
    pub fn remove_synapse(&mut self, id: GenIndex) -> bool {
        if !self.is_valid_synapse(id) {
            return false;
        }
        self.synapses[id.index].alive = false;
        self.free_synapse_slots.push(id.index);
        true
    }

    // =========================================================================
    // VALIDITY CHECKS
    // =========================================================================

    /// Returns `true` if `id` refers to a currently alive neuron.
    ///
    /// Checks:
    ///   1. Index is within arena bounds.
    ///   2. Generation counter matches (not a stale reference).
    ///   3. The neuron's alive flag is true.
    #[inline]
    pub fn is_valid_neuron(&self, id: GenIndex) -> bool {
        id.index < self.neurons.len()
            && self.neuron_generations[id.index] == id.generation
            && self.neurons[id.index].alive
    }

    /// Returns `true` if `id` refers to a currently alive synapse.
    #[inline]
    pub fn is_valid_synapse(&self, id: GenIndex) -> bool {
        id.index < self.synapses.len()
            && self.synapse_generations[id.index] == id.generation
            && self.synapses[id.index].alive
    }

    // =========================================================================
    // SAFE ACCESSORS
    // =========================================================================

    /// Returns an immutable reference to a neuron, or `None` if the id is stale.
    pub fn get_neuron(&self, id: GenIndex) -> Option<&Neuron> {
        if self.is_valid_neuron(id) {
            Some(&self.neurons[id.index])
        } else {
            None
        }
    }

    /// Returns a mutable reference to a neuron, or `None` if the id is stale.
    pub fn get_neuron_mut(&mut self, id: GenIndex) -> Option<&mut Neuron> {
        if self.is_valid_neuron(id) {
            Some(&mut self.neurons[id.index])
        } else {
            None
        }
    }

    /// Returns an immutable reference to a synapse, or `None` if the id is stale.
    pub fn get_synapse(&self, id: GenIndex) -> Option<&Synapse> {
        if self.is_valid_synapse(id) {
            Some(&self.synapses[id.index])
        } else {
            None
        }
    }

    /// Returns a mutable reference to a synapse, or `None` if the id is stale.
    pub fn get_synapse_mut(&mut self, id: GenIndex) -> Option<&mut Synapse> {
        if self.is_valid_synapse(id) {
            Some(&mut self.synapses[id.index])
        } else {
            None
        }
    }

    // =========================================================================
    // ITERATORS
    // =========================================================================

    /// Returns an iterator over all currently alive neurons.
    pub fn alive_neurons(&self) -> impl Iterator<Item = &Neuron> {
        self.neurons.iter().filter(|n| n.alive)
    }

    /// Returns an iterator over all currently alive neurons (mutable).
    pub fn alive_neurons_mut(&mut self) -> impl Iterator<Item = &mut Neuron> {
        self.neurons.iter_mut().filter(|n| n.alive)
    }

    /// Returns an iterator over all currently alive synapses.
    pub fn alive_synapses(&self) -> impl Iterator<Item = &Synapse> {
        self.synapses.iter().filter(|s| s.alive)
    }

    /// Returns an iterator over alive neurons filtered by role.
    pub fn neurons_by_role(&self, role: Role) -> impl Iterator<Item = &Neuron> {
        self.neurons
            .iter()
            .filter(move |n| n.alive && n.role == role)
    }

    // =========================================================================
    // DIAGNOSTICS
    // =========================================================================

    /// Returns the current count of alive neurons.
    pub fn alive_neuron_count(&self) -> usize {
        self.neurons.iter().filter(|n| n.alive).count()
    }

    /// Returns the current count of alive synapses.
    pub fn alive_synapse_count(&self) -> usize {
        self.synapses.iter().filter(|s| s.alive).count()
    }

    /// Returns the number of free neuron slots remaining.
    pub fn free_neuron_capacity(&self) -> usize {
        self.free_neuron_slots.len()
    }

    /// Returns the number of free synapse slots remaining.
    pub fn free_synapse_capacity(&self) -> usize {
        self.free_synapse_slots.len()
    }

    /// Resets all neuron activations to 0.0.
    ///
    /// Called at the beginning of every tick before injecting sensory input.
    ///
    /// Matches Python: NeuralGraph.reset_activations()
    pub fn reset_activations(&mut self) {
        for n in self.neurons.iter_mut() {
            if n.alive {
                n.activation = 0.0;
            }
        }
    }
}

impl Default for BrainMatrix {
    fn default() -> Self {
        Self::new()
    }
}
