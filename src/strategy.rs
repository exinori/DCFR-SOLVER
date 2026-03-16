/// Strategy extraction — convert Blueprint regrets to preflop charts.

use crate::game::{Action, Street};
use crate::infoset::InfoKey;
use crate::iso::{CanonicalHand, NUM_CANONICAL_HANDS};
use crate::mccfr::Blueprint;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PreflopChart
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct PreflopChart {
    pub position: String,          // e.g., "SB vs BB"
    pub hands: Vec<PreflopEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PreflopEntry {
    pub hand: String,          // e.g., "AKs"
    pub actions: Vec<ActionProb>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionProb {
    pub action: String,
    pub probability: f32,
}

impl PreflopChart {
    /// Extract preflop chart from blueprint.
    /// Returns the opening strategy (first action, no prior actions).
    pub fn from_blueprint(blueprint: &Blueprint) -> Self {
        let mut hands = Vec::new();

        for idx in 0..NUM_CANONICAL_HANDS as u8 {
            let ch = CanonicalHand::from_index(idx);
            let key = InfoKey::new(Street::Preflop, vec![], idx as u16);

            let entry = match blueprint.entries.get(&key) {
                Some(e) => e,
                None => continue,
            };

            let avg_strat = entry.average_strategy();

            // We need the actions for this node — they should match the betting grid
            // For preflop open (SB, depth=0): Fold, Call, Bet(2bb), Bet(3bb), Bet(4bb), Bet(8bb), AllIn
            // The number of actions in the entry tells us the action count.
            let action_labels = get_preflop_action_labels(entry.n_actions);

            let action_probs: Vec<ActionProb> = action_labels.iter()
                .zip(avg_strat.iter())
                .filter(|(_, &p)| p > 0.001)
                .map(|(label, &p)| ActionProb {
                    action: label.clone(),
                    probability: p,
                })
                .collect();

            if !action_probs.is_empty() {
                hands.push(PreflopEntry {
                    hand: ch.to_string(),
                    actions: action_probs,
                });
            }
        }

        PreflopChart {
            position: "SB open".to_string(),
            hands,
        }
    }
}

fn get_preflop_action_labels(n: usize) -> Vec<String> {
    // Default preflop open actions: fold, call, raise sizes..., allin
    // This matches the betting grid for preflop depth=0 facing BB
    let mut labels = vec!["fold".to_string(), "call".to_string()];
    let raise_labels = ["raise 2bb", "raise 3bb", "raise 4bb", "raise 8bb"];
    for i in 0..n.saturating_sub(3) {
        if i < raise_labels.len() {
            labels.push(raise_labels[i].to_string());
        }
    }
    labels.push("allin".to_string());

    // Trim to actual count
    labels.truncate(n);
    labels
}

/// Extract strategy for a specific action sequence.
pub fn extract_strategy(
    blueprint: &Blueprint,
    street: Street,
    action_seq: &[Action],
) -> HashMap<String, Vec<ActionProb>> {
    let mut result = HashMap::new();

    for idx in 0..NUM_CANONICAL_HANDS as u8 {
        let ch = CanonicalHand::from_index(idx);
        let key = InfoKey::new(street, action_seq.to_vec(), idx as u16);

        if let Some(entry) = blueprint.entries.get(&key) {
            let avg_strat = entry.average_strategy();
            let action_probs: Vec<ActionProb> = avg_strat.iter()
                .enumerate()
                .filter(|(_, &p)| p > 0.001)
                .map(|(i, &p)| ActionProb {
                    action: format!("action_{}", i),
                    probability: p,
                })
                .collect();

            if !action_probs.is_empty() {
                result.insert(ch.to_string(), action_probs);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_labels() {
        let labels = get_preflop_action_labels(7);
        assert_eq!(labels.len(), 7);
        assert_eq!(labels[0], "fold");
        assert_eq!(labels[1], "call");
    }
}
