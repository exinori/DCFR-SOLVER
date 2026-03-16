/// Blueprint information set key.
///
/// Groups hands by street + action sequence + abstraction bucket.
/// Used by MCCFR to aggregate regrets across similar hands.

use crate::game::{Action, Street};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InfoKey {
    pub street: Street,
    pub actions: Vec<Action>,   // action sequence this street only
    pub bucket: u16,            // preflop: 0..168, postflop: 0..49
    pub pot: i32,               // pot at start of current street (disambiguates action sets)
}

impl InfoKey {
    pub fn new(street: Street, actions: Vec<Action>, bucket: u16) -> Self {
        InfoKey { street, actions, bucket, pot: 0 }
    }

    pub fn with_pot(street: Street, actions: Vec<Action>, bucket: u16, pot: i32) -> Self {
        InfoKey { street, actions, bucket, pot }
    }
}

/// Regret entry for one information set.
pub struct RegretEntry {
    pub regrets: Vec<f32>,      // one per action
    pub cum_strategy: Vec<f32>, // cumulative strategy for averaging
    pub n_actions: usize,
}

impl RegretEntry {
    pub fn new(n_actions: usize) -> Self {
        RegretEntry {
            regrets: vec![0.0; n_actions],
            cum_strategy: vec![0.0; n_actions],
            n_actions,
        }
    }

    /// Current strategy from regret-matching (CFR+ style: floor at 0).
    pub fn current_strategy(&self) -> Vec<f32> {
        let positive_sum: f32 = self.regrets.iter().map(|&r| r.max(0.0)).sum();
        if positive_sum > 0.0 {
            self.regrets.iter().map(|&r| r.max(0.0) / positive_sum).collect()
        } else {
            vec![1.0 / self.n_actions as f32; self.n_actions]
        }
    }

    /// Average strategy (for final output).
    pub fn average_strategy(&self) -> Vec<f32> {
        let total: f32 = self.cum_strategy.iter().sum();
        if total > 0.0 {
            self.cum_strategy.iter().map(|&s| s / total).collect()
        } else {
            vec![1.0 / self.n_actions as f32; self.n_actions]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regret_entry_uniform() {
        let entry = RegretEntry::new(3);
        let strat = entry.current_strategy();
        assert!((strat[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((strat[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((strat[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_regret_matching() {
        let mut entry = RegretEntry::new(3);
        entry.regrets = vec![10.0, 5.0, 0.0];
        let strat = entry.current_strategy();
        assert!((strat[0] - 10.0 / 15.0).abs() < 1e-6);
        assert!((strat[1] - 5.0 / 15.0).abs() < 1e-6);
        assert!((strat[2] - 0.0).abs() < 1e-6);
    }
}
