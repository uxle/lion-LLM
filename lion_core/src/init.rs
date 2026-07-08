// lion_core/src/init.rs

use crate::brain::BrainMatrix;
use crate::constants::{
    NEURONS_PER_ROLE, PROCEDURAL_ACTIONS, SYNAPSE_CREATION_PROB,
};
use crate::rng::BrainRng;
use crate::types::{GenIndex, Role};

// =============================================================================
// BRAIN INITIALIZATION
// =============================================================================

impl BrainMatrix {
    /// Initializes the core brain with the canonical starting topology.
    ///
    /// Spawns exactly:
    ///   - 3 Vision neurons
    ///   - 3 Memory neurons
    ///   - 3 Danger neurons
    ///   - 1 Motor neuron per procedural action (WANDER, FORAGE, FLEE, ATTACK)
    ///
    /// Total: 13 neurons.
    ///
    /// Then runs `fully_connect_random()` to wire all neurons together
    /// with random sparse synapses.
    ///
    /// Matches Python:
    ///   def _initialize_core_brain(self):
    ///       for role in Role:
    ///           if role == Role.MOTOR:
    ///               for action in self.actions:
    ///                   self.add_neuron(NeUniform(role, self.dna, action_label=action))
    ///           else:
    ///               for _ in range(3):
    ///                   self.add_neuron(NeuronNode(role, self.dna))
    ///       self.fully_connect_random()
    ///
    /// # Panics
    /// Panics if the arena does not have enough free slots for the initial brain.
    /// With `MAX_NEURONS = 1024` and `INITIAL_NEURON_COUNT = 13`, this should never occur.
    pub fn initialize_core_brain(&mut self, rng: &mut BrainRng) {
        // Spawn non-motor neurons: 3 each for Vision, Memory, Danger.
        let non_motor_roles = [Role::Vision, Role::Memory, Role::Danger];

        for role in non_motor_roles {
            for _ in 0..NEURONS_PER_ROLE {
                let base_vector = rng.gen_base_vector();
                self.insert_neuron(role, base_vector)
                    .expect("Arena full during initialize_core_brain (non-motor)");
            }
        }

        // Spawn Motor neurons: one per procedural action.
        for &action in PROCEDURAL_ACTIONS {
            let base_vector = rng.gen_base_vector();
            self.insert_motor_neuron(base_vector, action)
                .expect("Arena full during initialize_core_brain (motor)");
        }

        // Wire all neurons with random sparse synapses.
        self.fully_connect_random(rng);
    }

    /// Creates a random sparse synapse graph over all currently alive neurons.
    ///
    /// For every ordered pair (pre, post) of distinct alive neurons,
    /// a synapse is created with probability `(1 - SYNAPSE_CREATION_PROB)`.
    ///
    /// Matches Python:
    ///   def fully_connect_random(self):
    ///       n_ids = list(self.neurons.keys())
    ///       for pre in n_ids:
    ///           for post in n_ids:
    ///               if pre != post and random.random() > 0.5:
    ///                   self.synapses[pre][post] = Synapse(pre, post, random.uniform(-0.5, 0.5))
    ///
    /// # Design note
    /// We cannot iterate over `self.neurons` and call `self.insert_synapse()`
    /// in the same loop because that would require a simultaneous mutable and
    /// immutable borrow of `self`. Instead, we collect all alive IDs into a
    /// temporary Vec first, then perform insertion in a separate pass.
    pub fn fully_connect_random(&mut self, rng: &mut BrainRng) {
        // Pass 1: collect all currently alive neuron GenIndex values.
        // This is a snapshot — safe to use across the mutable insertion pass.
        let ids: Vec<GenIndex> = self
            .neurons
            .iter()
            .filter(|n| n.alive)
            .map(|n| n.id)
            .collect();

        // Pass 2: for each ordered pair, probabilistically insert a synapse.
        for &pre in &ids {
            for &post in &ids {
                if pre == post {
                    continue;
                }
                // Matches Python: if random.random() > 0.5
                if rng.gen_prob() > SYNAPSE_CREATION_PROB {
                    let weight = rng.gen_initial_weight();
                    // insert_synapse validates both endpoints are still alive.
                    // If a slot is full it returns None — we silently skip.
                    let _ = self.insert_synapse(pre, post, weight);
                }
            }
        }
    }

    /// Returns the `GenIndex` of the Motor neuron whose action_label matches
    /// the given string, or `None` if no such neuron exists.
    ///
    /// Used during action extraction to locate the most activated Motor neuron.
    pub fn find_motor_neuron_by_label(&self, label: &str) -> Option<GenIndex> {
        self.neurons
            .iter()
            .find(|n| {
                n.alive
                    && n.role == crate::types::Role::Motor
                    && n.action_label
                        .map(|l| l.as_str() == label)
                        .unwrap_or(false)
            })
            .map(|n| n.id)
    }

    /// Returns a snapshot Vec of all alive neuron GenIndex values.
    ///
    /// Use whenever you need to iterate over neuron IDs while also mutating
    /// the arena — collect first, then mutate.
    ///
    /// This pattern replaces Python's: `n_ids = list(self.neurons.keys())`
    pub fn collect_alive_neuron_ids(&self) -> Vec<GenIndex> {
        self.neurons
            .iter()
            .filter(|n| n.alive)
            .map(|n| n.id)
            .collect()
    }

    /// Returns a snapshot Vec of all alive synapse GenIndex values.
    ///
    /// Same pattern as `collect_alive_neuron_ids` — collect before mutating.
    pub fn collect_alive_synapse_ids(&self) -> Vec<GenIndex> {
        self.synapses
            .iter()
            .enumerate()
            .filter(|(_, s)| s.alive)
            .map(|(i, _)| GenIndex::new(i, self.synapse_generations[i]))
            .collect()
    }
}
