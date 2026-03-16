/// External Sampling MCCFR for Blueprint strategy computation.
///
/// Uses equity abstraction to group hands into buckets.
/// Traverser explores all actions, opponent samples one action.

use crate::abstraction::EquityAbstraction;
use crate::card::{Card, Hand, NUM_CARDS};
use crate::deck::Deck;
use crate::game::{Action, GameState, NodeType, Player, Street, OOP, IP};
use crate::infoset::{InfoKey, RegretEntry};
use crate::iso::canonical_hand;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Blueprint
// ---------------------------------------------------------------------------

pub struct Blueprint {
    pub entries: HashMap<InfoKey, RegretEntry>,
    pub iterations: u64,
}

impl Blueprint {
    pub fn new() -> Self {
        Blueprint {
            entries: HashMap::new(),
            iterations: 0,
        }
    }

    fn get_or_create(&mut self, key: &InfoKey, n_actions: usize) -> &mut RegretEntry {
        if !self.entries.contains_key(key) {
            self.entries.insert(key.clone(), RegretEntry::new(n_actions));
        }
        self.entries.get_mut(key).unwrap()
    }
}

// ---------------------------------------------------------------------------
// McfrTrainer
// ---------------------------------------------------------------------------

pub struct McfrTrainer {
    pub blueprint: Blueprint,
    pub abstraction: EquityAbstraction,
    rng: SmallRng,
}

impl McfrTrainer {
    pub fn new(abstraction: EquityAbstraction) -> Self {
        McfrTrainer {
            blueprint: Blueprint::new(),
            abstraction,
            rng: SmallRng::seed_from_u64(42),
        }
    }

    pub fn new_with_seed(abstraction: EquityAbstraction, seed: u64) -> Self {
        McfrTrainer {
            blueprint: Blueprint::new(),
            abstraction,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Run MCCFR for n iterations.
    pub fn train(&mut self, iterations: u64, stack: i32) {
        for i in 0..iterations {
            if i % 100_000 == 0 && i > 0 {
                println!("  iteration {}/{}", i, iterations);
            }

            // Deal random hand
            let mut deck = Deck::new_from_rng(&mut self.rng);
            let state = GameState::new_hand(stack);
            let hole0 = deck.draw_hand(2);
            let hole1 = deck.draw_hand(2);
            let state = state.set_holes(OOP, hole0).set_holes(IP, hole1);

            // Alternate traverser
            let traverser = if i % 2 == 0 { OOP } else { IP };

            self.cfr_external(
                &state,
                traverser,
                &mut deck,
                &[],
                state.pot,
            );

            self.blueprint.iterations += 1;
        }
    }

    /// External sampling MCCFR recursive traversal.
    fn cfr_external(
        &mut self,
        state: &GameState,
        traverser: Player,
        deck: &mut Deck,
        action_seq: &[Action],
        street_pot: i32,
    ) -> f32 {
        match state.node_type() {
            NodeType::Terminal(_) => {
                state.payoff(traverser)
            }
            NodeType::Chance(next_street) => {
                // Sample random cards for next street
                let dead = state.board.union(state.holes[0]).union(state.holes[1]);
                let n_cards = match next_street {
                    Street::Flop => 3,
                    _ => 1,
                };
                let cards = sample_cards(&mut self.rng, dead, n_cards);
                let next_state = state.deal_street(&cards);
                // Reset action seq for new street, pass new pot
                self.cfr_external(&next_state, traverser, deck, &[], next_state.pot)
            }
            NodeType::Decision(player) => {
                let actions = state.actions();
                let n_actions = actions.len();

                // Compute bucket for current player
                let hole = state.holes[player as usize];
                let bucket = self.get_bucket(hole, state.board, state.street);

                let key = InfoKey::with_pot(state.street, action_seq.to_vec(), bucket, street_pot);

                let strategy = {
                    let entry = self.blueprint.get_or_create(&key, n_actions);
                    entry.current_strategy()
                };

                if player == traverser {
                    // Explore all actions
                    let mut action_values = vec![0.0f32; n_actions];
                    let mut node_value = 0.0f32;

                    for (a, action) in actions.iter().enumerate() {
                        let next_state = state.apply(*action);
                        let mut next_seq = action_seq.to_vec();
                        next_seq.push(*action);
                        let value = self.cfr_external(&next_state, traverser, deck, &next_seq, street_pot);
                        action_values[a] = value;
                        node_value += strategy[a] * value;
                    }

                    // Update regrets (CFR+: floor at 0)
                    let weight = self.blueprint.iterations as f32 + 1.0; // Linear CFR weighting
                    let entry = self.blueprint.get_or_create(&key, n_actions);
                    for a in 0..n_actions {
                        let regret = action_values[a] - node_value;
                        entry.regrets[a] = (entry.regrets[a] + regret).max(0.0);
                    }
                    // Accumulate strategy with linear weighting
                    for a in 0..n_actions {
                        entry.cum_strategy[a] += weight * strategy[a];
                    }

                    node_value
                } else {
                    // Sample one action from current strategy
                    let action_idx = sample_action(&strategy, &mut self.rng);
                    let next_state = state.apply(actions[action_idx]);
                    let mut next_seq = action_seq.to_vec();
                    next_seq.push(actions[action_idx]);
                    self.cfr_external(&next_state, traverser, deck, &next_seq, street_pot)
                }
            }
        }
    }

    fn get_bucket(&self, hole: Hand, board: Hand, street: Street) -> u16 {
        match street {
            Street::Preflop => {
                let cards: Vec<Card> = hole.iter().collect();
                let ch = canonical_hand(cards[0], cards[1]);
                ch.index() as u16
            }
            _ => {
                self.abstraction.bucket(hole, board, street) as u16
            }
        }
    }
}

fn sample_cards(rng: &mut SmallRng, dead: Hand, n: usize) -> Vec<Card> {
    let mut available: Vec<Card> = (0..NUM_CARDS as Card)
        .filter(|&c| !dead.contains(c))
        .collect();

    let mut result = Vec::with_capacity(n);
    for _i in 0..n {
        let idx = rng.gen_range(0..available.len());
        result.push(available.swap_remove(idx));
    }
    result
}

fn sample_action(strategy: &[f32], rng: &mut SmallRng) -> usize {
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in strategy.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    strategy.len() - 1
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

impl Blueprint {
    pub fn save<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};
        w.write_u64::<LittleEndian>(self.iterations)?;
        w.write_u64::<LittleEndian>(self.entries.len() as u64)?;

        for (key, entry) in &self.entries {
            // Write key
            w.write_u8(key.street as u8)?;
            w.write_u16::<LittleEndian>(key.bucket)?;
            w.write_i32::<LittleEndian>(key.pot)?;
            w.write_u16::<LittleEndian>(key.actions.len() as u16)?;
            for action in &key.actions {
                write_action(w, *action)?;
            }

            // Write entry
            w.write_u16::<LittleEndian>(entry.n_actions as u16)?;
            for &r in &entry.regrets {
                w.write_f32::<LittleEndian>(r)?;
            }
            for &s in &entry.cum_strategy {
                w.write_f32::<LittleEndian>(s)?;
            }
        }

        Ok(())
    }

    pub fn load<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        let iterations = r.read_u64::<LittleEndian>()?;
        let n_entries = r.read_u64::<LittleEndian>()? as usize;

        let mut entries = HashMap::with_capacity(n_entries);
        for _ in 0..n_entries {
            let street_byte = r.read_u8()?;
            let street = match street_byte {
                0 => Street::Preflop,
                1 => Street::Flop,
                2 => Street::Turn,
                3 => Street::River,
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad street")),
            };
            let bucket = r.read_u16::<LittleEndian>()?;
            let pot = r.read_i32::<LittleEndian>()?;
            let n_actions_in_seq = r.read_u16::<LittleEndian>()? as usize;
            let mut actions = Vec::with_capacity(n_actions_in_seq);
            for _ in 0..n_actions_in_seq {
                actions.push(read_action(r)?);
            }

            let key = InfoKey::with_pot(street, actions, bucket, pot);

            let n_actions = r.read_u16::<LittleEndian>()? as usize;
            let mut regrets = vec![0.0f32; n_actions];
            for i in 0..n_actions {
                regrets[i] = r.read_f32::<LittleEndian>()?;
            }
            let mut cum_strategy = vec![0.0f32; n_actions];
            for i in 0..n_actions {
                cum_strategy[i] = r.read_f32::<LittleEndian>()?;
            }

            entries.insert(key, RegretEntry {
                regrets,
                cum_strategy,
                n_actions,
            });
        }

        Ok(Blueprint { entries, iterations })
    }
}

fn write_action<W: std::io::Write>(w: &mut W, action: Action) -> std::io::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use crate::game::BetSize;
    match action {
        Action::Fold => w.write_u8(0)?,
        Action::Check => w.write_u8(1)?,
        Action::Call => w.write_u8(2)?,
        Action::AllIn => w.write_u8(3)?,
        Action::Bet(BetSize::Frac(n, d)) => {
            w.write_u8(4)?;
            w.write_i32::<LittleEndian>(n)?;
            w.write_i32::<LittleEndian>(d)?;
        }
        Action::Bet(BetSize::Bb(bb)) => {
            w.write_u8(5)?;
            w.write_i32::<LittleEndian>(bb)?;
        }
    }
    Ok(())
}

fn read_action<R: std::io::Read>(r: &mut R) -> std::io::Result<Action> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use crate::game::BetSize;
    let tag = r.read_u8()?;
    match tag {
        0 => Ok(Action::Fold),
        1 => Ok(Action::Check),
        2 => Ok(Action::Call),
        3 => Ok(Action::AllIn),
        4 => {
            let n = r.read_i32::<LittleEndian>()?;
            let d = r.read_i32::<LittleEndian>()?;
            Ok(Action::Bet(BetSize::Frac(n, d)))
        }
        5 => {
            let bb = r.read_i32::<LittleEndian>()?;
            Ok(Action::Bet(BetSize::Bb(bb)))
        }
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad action")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blueprint_creation() {
        let abstraction = EquityAbstraction::new();
        let mut trainer = McfrTrainer::new(abstraction);
        // Run a few iterations (preflop only, no postflop abstraction)
        trainer.train(100, 100);
        assert_eq!(trainer.blueprint.iterations, 100);
        assert!(trainer.blueprint.entries.len() > 0);
    }

    #[test]
    fn test_blueprint_serialize_roundtrip() {
        let abstraction = EquityAbstraction::new();
        let mut trainer = McfrTrainer::new(abstraction);
        trainer.train(50, 100);

        // Serialize
        let mut buf = Vec::new();
        trainer.blueprint.save(&mut buf).unwrap();

        // Deserialize
        let loaded = Blueprint::load(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.iterations, 50);
        assert_eq!(loaded.entries.len(), trainer.blueprint.entries.len());
    }
}
