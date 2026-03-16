/// Kuhn poker — minimal 3-card game for CFR+ algorithm verification.
///
/// Rules:
///   - 3 cards: J(0), Q(1), K(2)
///   - Each player antes 1
///   - Player 0 acts first: Check or Bet(1)
///   - If Check → Player 1: Check(showdown) or Bet(1)
///     - If Bet → Player 0: Call(1) or Fold
///   - If Bet → Player 1: Call(1) or Fold
///
/// Known Nash equilibrium (game value = -1/18 for Player 0):
///
/// Player 0 (parameterized by α ∈ [0, 1/3]):
///   J: bet α, check (1-α);  check→bet: fold
///   Q: check;               check→bet: call (α + 1/3)
///   K: bet 3α, check (1-3α); check→bet: call
///
/// Player 1:
///   J: check→check; bet→fold;  (after P0 check: bet 1/3)
///   Q: check→check; bet→call 1/3
///   K: check→bet;   bet→call
///
/// At α=0: P0 never bluffs with J, bets K 0% (always checks), Q calls 1/3.
/// CFR typically converges to exploitability → 0, game value → -1/18 ≈ -0.0556

use std::collections::HashMap;

const NUM_CARDS: usize = 3; // J=0, Q=1, K=2
const CARD_NAMES: [&str; 3] = ["J", "Q", "K"];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InfoSet {
    card: usize,       // 0=J, 1=Q, 2=K
    history: String,   // sequence of actions: "cb" = check then bet, etc.
}

struct Node {
    regret_sum: Vec<f32>,
    strategy_sum: Vec<f32>,
    n_actions: usize,
}

impl Node {
    fn new(n_actions: usize) -> Self {
        Node {
            regret_sum: vec![0.0; n_actions],
            strategy_sum: vec![0.0; n_actions],
            n_actions,
        }
    }

    fn current_strategy(&self) -> Vec<f32> {
        let mut strat = vec![0.0f32; self.n_actions];
        let mut normalizing_sum = 0.0f32;

        for a in 0..self.n_actions {
            strat[a] = self.regret_sum[a].max(0.0);
            normalizing_sum += strat[a];
        }

        if normalizing_sum > 0.0 {
            for s in &mut strat {
                *s /= normalizing_sum;
            }
        } else {
            let uniform = 1.0 / self.n_actions as f32;
            strat.fill(uniform);
        }

        strat
    }

    fn average_strategy(&self) -> Vec<f32> {
        let total: f32 = self.strategy_sum.iter().sum();
        if total > 0.0 {
            self.strategy_sum.iter().map(|&s| s / total).collect()
        } else {
            vec![1.0 / self.n_actions as f32; self.n_actions]
        }
    }
}

pub struct KuhnCfr {
    nodes: HashMap<InfoSet, Node>,
    iteration: u32,
}

impl KuhnCfr {
    pub fn new() -> Self {
        KuhnCfr {
            nodes: HashMap::new(),
            iteration: 0,
        }
    }

    /// Run CFR+ with linear weighting for given iterations.
    /// Returns average game value for P0.
    pub fn train(&mut self, iterations: u32) -> f32 {
        let mut util_sum = 0.0f32;

        for _ in 0..iterations {
            self.iteration += 1;
            for c0 in 0..NUM_CARDS {
                for c1 in 0..NUM_CARDS {
                    if c0 == c1 { continue; }
                    let cards = [c0, c1];
                    let val = self.cfr(&cards, String::new(), 1.0, 1.0);
                    util_sum += val;
                }
            }
        }

        util_sum / (iterations as f32 * 6.0)
    }

    /// CFR+ recursive traversal.
    /// Returns utility for Player 0.
    fn cfr(
        &mut self,
        cards: &[usize; 2],
        history: String,
        pi0: f32,
        pi1: f32,
    ) -> f32 {
        if let Some(payoff) = terminal_payoff(cards, &history) {
            return payoff;
        }

        let player = current_player(&history);

        let info = InfoSet {
            card: cards[player],
            history: history.clone(),
        };

        let actions = legal_actions(&history);
        let n_actions = actions.len();

        if !self.nodes.contains_key(&info) {
            self.nodes.insert(info.clone(), Node::new(n_actions));
        }

        let strategy = self.nodes.get(&info).unwrap().current_strategy();

        let mut action_utils = vec![0.0f32; n_actions];
        let mut node_util = 0.0f32;

        for (a, action) in actions.iter().enumerate() {
            let next_history = format!("{}{}", history, action);
            let action_util = if player == 0 {
                self.cfr(cards, next_history, pi0 * strategy[a], pi1)
            } else {
                self.cfr(cards, next_history, pi0, pi1 * strategy[a])
            };
            action_utils[a] = action_util;
            node_util += strategy[a] * action_util;
        }

        let node = self.nodes.get_mut(&info).unwrap();
        let opp_reach = if player == 0 { pi1 } else { pi0 };
        let sign = if player == 0 { 1.0 } else { -1.0 };

        // CFR+: update regrets with floor at 0
        for a in 0..n_actions {
            let regret = sign * (action_utils[a] - node_util);
            node.regret_sum[a] = (node.regret_sum[a] + opp_reach * regret).max(0.0);
        }

        // Linear CFR: weight strategy by iteration number
        let own_reach = if player == 0 { pi0 } else { pi1 };
        let weight = self.iteration as f32;
        for a in 0..n_actions {
            node.strategy_sum[a] += weight * own_reach * strategy[a];
        }

        node_util
    }

    /// Compute exploitability via complete enumeration.
    /// For each player, compute best response EV against opponent's avg strategy.
    /// Uses full tree enumeration — correct for small games like Kuhn.
    pub fn exploitability(&self) -> f32 {
        let br0 = self.br_full_enum(0);
        let br1 = self.br_full_enum(1);

        #[cfg(test)]
        println!("  BR_P0={:.6}, BR_P1={:.6}", br0, br1);

        br0 + br1
    }

    /// Full enumeration best response for br_player.
    /// For each BR info set, try all pure strategies and pick the best one.
    fn br_full_enum(&self, br_player: usize) -> f32 {
        // Collect all possible BR info sets (card + history where it's BR player's turn)
        let br_histories: Vec<&str> = if br_player == 0 {
            vec!["", "cb"]  // P0 acts at "" and "cb"
        } else {
            vec!["c", "b"]  // P1 acts at "c" and "b"
        };

        // Each info set = (card, history), with 2 actions each
        // Total BR strategies: 2^(3*2) = 64 for each player (3 cards × 2 histories × 2 actions)
        // Just enumerate all 64 pure strategies and pick the best

        let n_infosets = NUM_CARDS * br_histories.len(); // 6
        let n_strategies = 1u32 << n_infosets; // 64

        let mut best_ev = f32::NEG_INFINITY;

        for strat_bits in 0..n_strategies {
            // Build pure strategy: for info set i, action = bit i of strat_bits
            let br_strat: Vec<usize> = (0..n_infosets)
                .map(|i| ((strat_bits >> i) & 1) as usize)
                .collect();

            // Evaluate this strategy against opponent's average strategy
            let mut total_ev = 0.0f32;
            let mut n_deals = 0;

            for c0 in 0..NUM_CARDS {
                for c1 in 0..NUM_CARDS {
                    if c0 == c1 { continue; }
                    let cards = [c0, c1];
                    let ev = self.eval_br_strategy(&cards, "", br_player, &br_strat, &br_histories);
                    total_ev += ev;
                    n_deals += 1;
                }
            }

            let avg_ev = total_ev / n_deals as f32;
            best_ev = best_ev.max(avg_ev);
        }

        best_ev
    }

    /// Evaluate a specific pure BR strategy on a specific deal.
    fn eval_br_strategy(
        &self,
        cards: &[usize; 2],
        history: &str,
        br_player: usize,
        br_strat: &[usize],
        br_histories: &[&str],
    ) -> f32 {
        if let Some(payoff) = terminal_payoff(cards, history) {
            return if br_player == 0 { payoff } else { -payoff };
        }

        let player = current_player(history);
        let actions = legal_actions(history);

        if player == br_player {
            // Use the pure BR strategy
            let my_card = cards[br_player];
            let hist_idx = br_histories.iter().position(|&h| h == history).unwrap();
            let infoset_idx = my_card * br_histories.len() + hist_idx;
            let action_idx = br_strat[infoset_idx];
            let action_idx = action_idx.min(actions.len() - 1);
            let next = format!("{}{}", history, actions[action_idx]);
            self.eval_br_strategy(cards, &next, br_player, br_strat, br_histories)
        } else {
            // Opponent uses average strategy
            let info = InfoSet {
                card: cards[player],
                history: history.to_string(),
            };
            let strat = match self.nodes.get(&info) {
                Some(node) => node.average_strategy(),
                None => vec![1.0 / actions.len() as f32; actions.len()],
            };
            let mut val = 0.0f32;
            for (a, action) in actions.iter().enumerate() {
                let next = format!("{}{}", history, action);
                val += strat[a] * self.eval_br_strategy(cards, &next, br_player, br_strat, br_histories);
            }
            val
        }
    }

    /// Print the average strategy for all info sets.
    pub fn print_strategy(&self) {
        let mut keys: Vec<&InfoSet> = self.nodes.keys().collect();
        keys.sort_by(|a, b| {
            a.history.len().cmp(&b.history.len())
                .then(a.card.cmp(&b.card))
                .then(a.history.cmp(&b.history))
        });

        let action_labels = |h: &str| -> Vec<&str> {
            let actions = legal_actions(h);
            actions.iter().map(|a| match a {
                'c' => "check",
                'b' => "bet",
                'k' => "call",
                'f' => "fold",
                _ => "?",
            }).collect()
        };

        for info in keys {
            let node = self.nodes.get(info).unwrap();
            let avg = node.average_strategy();
            let labels = action_labels(&info.history);
            let player = current_player(&info.history);

            print!("P{} {} [{}]: ", player, CARD_NAMES[info.card], info.history);
            for (i, &p) in avg.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{}={:.3}", labels[i], p);
            }
            println!();
        }
    }
}

// ---------------------------------------------------------------------------
// Kuhn poker game logic
// ---------------------------------------------------------------------------

fn current_player(history: &str) -> usize {
    // P0 acts at "", "cb" (check→bet, P0 responds)
    // P1 acts at "c" (after check), "b" (after bet)
    match history {
        "" => 0,
        "c" => 1,
        "b" => 1,
        "cb" => 0,
        _ => unreachable!("invalid history: {}", history),
    }
}

fn legal_actions(history: &str) -> Vec<char> {
    match history {
        "" => vec!['c', 'b'],       // P0: check or bet
        "c" => vec!['c', 'b'],      // P1: check or bet
        "b" => vec!['f', 'k'],      // P1: fold or call
        "cb" => vec!['f', 'k'],     // P0: fold or call
        _ => unreachable!("invalid history for actions: {}", history),
    }
}

/// Terminal payoff for Player 0 (positive = P0 wins).
fn terminal_payoff(cards: &[usize; 2], history: &str) -> Option<f32> {
    let winner = if cards[0] > cards[1] { 0 } else { 1 };

    match history {
        "cc" => {
            // Check-check: showdown, pot = 2 (1+1 ante)
            if winner == 0 { Some(1.0) } else { Some(-1.0) }
        }
        "bf" => {
            // P0 bet, P1 fold: P0 wins ante (1)
            Some(1.0)
        }
        "bk" => {
            // P0 bet, P1 call: showdown, pot = 4 (1+1 ante + 1+1 bet)
            if winner == 0 { Some(2.0) } else { Some(-2.0) }
        }
        "cbf" => {
            // P0 check, P1 bet, P0 fold: P1 wins ante (1)
            Some(-1.0)
        }
        "cbk" => {
            // P0 check, P1 bet, P0 call: showdown, pot = 4
            if winner == 0 { Some(2.0) } else { Some(-2.0) }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kuhn_terminal_payoffs() {
        // K vs J, check-check: K wins 1
        assert_eq!(terminal_payoff(&[2, 0], "cc"), Some(1.0));
        // J vs K, check-check: J loses 1
        assert_eq!(terminal_payoff(&[0, 2], "cc"), Some(-1.0));
        // bet-fold: bettor always wins 1
        assert_eq!(terminal_payoff(&[0, 2], "bf"), Some(1.0));
        // bet-call with K vs J: K wins 2
        assert_eq!(terminal_payoff(&[2, 0], "bk"), Some(2.0));
    }

    #[test]
    fn test_kuhn_convergence() {
        let mut cfr = KuhnCfr::new();
        let game_value = cfr.train(100_000);

        // Nash equilibrium game value = -1/18 ≈ -0.0556
        let expected = -1.0 / 18.0;
        println!("Game value: {:.6} (expected {:.6})", game_value, expected);
        assert!(
            (game_value - expected).abs() < 0.01,
            "Game value {:.6} not close to {:.6}",
            game_value, expected
        );

        // Check exploitability → close to 0
        let exploit = cfr.exploitability();
        println!("Exploitability: {:.6}", exploit);
        assert!(
            exploit < 0.02,
            "Exploitability {:.6} too high (should be < 0.02)",
            exploit
        );

        println!("\nFinal average strategy:");
        cfr.print_strategy();

        // Verify specific strategy properties:
        // P1 with K facing bet: should always call
        let k_facing_bet = cfr.nodes.get(&InfoSet {
            card: 2,
            history: "b".to_string(),
        }).unwrap().average_strategy();
        assert!(
            k_facing_bet[1] > 0.95,
            "P1 K facing bet should call ≈1.0, got {:.3}",
            k_facing_bet[1]
        );

        // P1 with J facing bet: should always fold
        let j_facing_bet = cfr.nodes.get(&InfoSet {
            card: 0,
            history: "b".to_string(),
        }).unwrap().average_strategy();
        assert!(
            j_facing_bet[0] > 0.95,
            "P1 J facing bet should fold ≈1.0, got {:.3}",
            j_facing_bet[0]
        );

        // P0 with K after check-bet: should always call
        let k_check_bet = cfr.nodes.get(&InfoSet {
            card: 2,
            history: "cb".to_string(),
        }).unwrap().average_strategy();
        assert!(
            k_check_bet[1] > 0.95,
            "P0 K after check-bet should call ≈1.0, got {:.3}",
            k_check_bet[1]
        );

        // P0 with J after check-bet: should always fold
        let j_check_bet = cfr.nodes.get(&InfoSet {
            card: 0,
            history: "cb".to_string(),
        }).unwrap().average_strategy();
        assert!(
            j_check_bet[0] > 0.95,
            "P0 J after check-bet should fold ≈1.0, got {:.3}",
            j_check_bet[0]
        );

        // P1 with Q facing bet: should call ≈1/3
        let q_facing_bet = cfr.nodes.get(&InfoSet {
            card: 1,
            history: "b".to_string(),
        }).unwrap().average_strategy();
        assert!(
            (q_facing_bet[1] - 1.0 / 3.0).abs() < 0.05,
            "P1 Q facing bet should call ≈1/3, got {:.3}",
            q_facing_bet[1]
        );

        // P1 with J after check: should bet ≈1/3
        let j_after_check = cfr.nodes.get(&InfoSet {
            card: 0,
            history: "c".to_string(),
        }).unwrap().average_strategy();
        assert!(
            (j_after_check[1] - 1.0 / 3.0).abs() < 0.05,
            "P1 J after check should bet ≈1/3, got {:.3}",
            j_after_check[1]
        );
    }

    #[test]
    fn test_kuhn_convergence_detailed() {
        let mut cfr = KuhnCfr::new();
        let gv = cfr.train(200_000);
        let exploit = cfr.exploitability();
        println!("Game value: {:.6} (expected {:.6})", gv, -1.0 / 18.0);
        println!("Exploitability: {:.6}", exploit);
        cfr.print_strategy();

        // Game value check
        assert!((gv - (-1.0/18.0)).abs() < 0.005, "game value");
        // Exploitability check
        assert!(exploit < 0.005, "exploitability {:.6} too high", exploit);
    }
}
