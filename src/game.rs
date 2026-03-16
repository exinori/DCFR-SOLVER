/// NLHE GameState — heads-up no-limit hold'em with configurable betting grid.
///
/// Supports both Blueprint (MCCFR) and Subgame (CFR+) engines via shared game logic.

use crate::card::{Card, Hand};
use crate::eval::evaluate;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub type Player = u8;
pub const OOP: Player = 0; // out of position (SB preflop, first postflop)
pub const IP: Player = 1; // in position (BB preflop, last postflop)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

impl Street {
    pub fn next(self) -> Option<Street> {
        match self {
            Street::Preflop => Some(Street::Flop),
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => None,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 1,
            Street::Turn => 2,
            Street::River => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BetSize {
    // Fractions of pot (numerator, denominator)
    Frac(i32, i32),
    // Fixed BB amount (preflop opens)
    Bb(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(BetSize),
    AllIn,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Fold => write!(f, "fold"),
            Action::Check => write!(f, "check"),
            Action::Call => write!(f, "call"),
            Action::Bet(BetSize::Frac(n, d)) => {
                if *d == 1 {
                    write!(f, "bet {}x", n)
                } else if *d == 100 {
                    write!(f, "bet {}%", n)
                } else {
                    write!(f, "bet {}/{}", n, d)
                }
            }
            Action::Bet(BetSize::Bb(bb)) => write!(f, "bet {}bb", bb),
            Action::AllIn => write!(f, "allin"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalType {
    Fold(Player),   // who folded
    Showdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Terminal(TerminalType),
    Chance(Street),         // deal next street
    Decision(Player),
}

// ---------------------------------------------------------------------------
// Betting Grid
// ---------------------------------------------------------------------------

const DEFAULT_MAX_RAISES: u8 = 3;

// ---------------------------------------------------------------------------
// BetConfig — configurable bet sizing
// ---------------------------------------------------------------------------

/// Configurable bet sizes for the game tree.
/// For each street, specify sizes at each depth (0 = first bet, 1 = raise, 2+ = reraise).
#[derive(Clone, Debug)]
pub struct BetConfig {
    /// Bet sizes per street. Each Vec<Vec<BetSize>> is indexed by depth.
    /// sizes[street_idx][depth] → list of bet sizes.
    /// street_idx: 0=preflop, 1=flop, 2=turn, 3=river
    pub sizes: [Vec<Vec<BetSize>>; 4],
    pub max_raises: u8,
    /// All-in threshold: if a bet/raise uses >= this fraction of the player's
    /// remaining stack, collapse it to all-in (skip the intermediate size).
    /// 0.0 = disabled (no collapse), 0.67 = collapse when using >= 67% of stack.
    pub allin_threshold: f32,
    /// All-in pot ratio: only add the AllIn action when the player's remaining
    /// stack ≤ current_pot × this ratio. 0.0 = always allow all-in (default).
    /// E.g., 3.0 = all-in only when stack ≤ 3× pot (SPR ≤ 3).
    pub allin_pot_ratio: f32,
    /// Disable donk betting: OOP can only check when first to act (no bet/allin).
    /// Matches GTO+ "Donk OFF" setting.
    pub no_donk: bool,
    /// Geometric sizing: when only 2 bets remain before cap, replace configured
    /// bet sizes with the geometric size that reaches all-in in exactly 2 bets.
    /// Matches GTO+ "With only 2 bets left, use geometric sizing".
    pub geometric_2bets: bool,
}

impl Default for BetConfig {
    fn default() -> Self {
        BetConfig {
            sizes: [
                // Preflop
                vec![
                    vec![BetSize::Bb(2), BetSize::Bb(3), BetSize::Bb(4), BetSize::Bb(8)],
                    vec![BetSize::Frac(1, 1), BetSize::Frac(3, 2), BetSize::Frac(2, 1)],
                    vec![BetSize::Frac(1, 1), BetSize::Frac(2, 1)],
                ],
                // Flop (33% = 33/100, not 1/3 = 33.33%)
                vec![
                    vec![BetSize::Frac(33, 100), BetSize::Frac(1, 2), BetSize::Frac(1, 1), BetSize::Frac(2, 1)],
                    vec![BetSize::Frac(67, 100), BetSize::Frac(1, 1), BetSize::Frac(3, 2)],
                    vec![BetSize::Frac(1, 1), BetSize::Frac(3, 2)],
                ],
                // Turn
                vec![
                    vec![BetSize::Frac(33, 100), BetSize::Frac(67, 100), BetSize::Frac(1, 1), BetSize::Frac(2, 1)],
                    vec![BetSize::Frac(1, 1), BetSize::Frac(3, 2)],
                ],
                // River
                vec![
                    vec![BetSize::Frac(33, 100), BetSize::Frac(1, 2), BetSize::Frac(1, 1), BetSize::Frac(2, 1)],
                    vec![BetSize::Frac(67, 100), BetSize::Frac(1, 1), BetSize::Frac(2, 1)],
                    vec![BetSize::Frac(1, 1)],
                ],
            ],
            max_raises: DEFAULT_MAX_RAISES,
            allin_threshold: 0.0,
            allin_pot_ratio: 0.0,
            no_donk: false,
            geometric_2bets: false,
        }
    }
}

impl BetConfig {
    /// Get bet sizes for a given street and depth.
    pub fn bet_sizes(&self, street: Street, depth: u8) -> &[BetSize] {
        let idx = street.index();
        let depths = &self.sizes[idx];
        let d = (depth as usize).min(depths.len().saturating_sub(1));
        if d < depths.len() {
            &depths[d]
        } else {
            &[]
        }
    }
}

/// Returns available bet sizes for the current situation (default config).
fn default_bet_sizes(street: Street, depth: u8) -> &'static [BetSize] {
    static DEFAULT: std::sync::OnceLock<BetConfig> = std::sync::OnceLock::new();
    let config = DEFAULT.get_or_init(BetConfig::default);
    // SAFETY: BetConfig::default() creates owned data that lives in OnceLock
    // We can return a reference because OnceLock data is 'static.
    config.bet_sizes(street, depth)
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GameState {
    pub street: Street,
    pub pot: i32,              // total pot (excluding current bets)
    pub stacks: [i32; 2],     // remaining stack behind
    pub bets: [i32; 2],       // current street bets
    pub board: Hand,
    pub holes: [Hand; 2],
    pub to_act: Player,
    pub n_raises: u8,         // raises this street
    pub closed: bool,         // both players have acted, action closed
    pub is_allin: [bool; 2],
    pub actions_this_street: u8,
    pub last_aggressor: Option<Player>,
    pub folded: Option<Player>,         // who folded (if fold terminal)
    pub initial_stacks: [i32; 2],      // starting stacks for invested() calculation
    pub bet_config: Option<Arc<BetConfig>>,  // None = use defaults
}

impl GameState {
    /// Standard 6-max starting state (100bb effective, SB=0.5bb, BB=1bb).
    /// SB (OOP) posts 1bb, BB (IP) posts 2bb. SB acts first preflop.
    /// Using integer chips: 1 chip = 0.5bb, so SB=1, BB=2, stacks=200.
    /// Actually let's use bb directly with i32: SB=1, BB=2, stack=100.
    pub fn new_hand(stack: i32) -> Self {
        let stacks = [stack - 1, stack - 2];
        GameState {
            street: Street::Preflop,
            pot: 0,
            stacks,
            bets: [1, 2],                    // SB=1bb, BB=2bb
            board: Hand::new(),
            holes: [Hand::new(), Hand::new()],
            to_act: OOP, // SB acts first preflop
            n_raises: 1, // BB is the initial "raise" (open counts as depth=0)
            closed: false,
            is_allin: [false, false],
            actions_this_street: 0,
            last_aggressor: None,
            folded: None,
            initial_stacks: [stack, stack], // pre-blind stacks
            bet_config: None,
        }
    }

    /// Create a postflop state for subgame solving.
    pub fn new_postflop(
        street: Street,
        pot: i32,
        stacks: [i32; 2],
        board: Hand,
    ) -> Self {
        GameState {
            street,
            pot,
            stacks,
            bets: [0, 0],
            board,
            holes: [Hand::new(), Hand::new()],
            to_act: OOP, // OOP acts first postflop
            n_raises: 0,
            closed: false,
            is_allin: [false, false],
            actions_this_street: 0,
            last_aggressor: None,
            folded: None,
            initial_stacks: stacks, // stacks at subgame start
            bet_config: None,
        }
    }

    /// Create a postflop state with custom bet config.
    pub fn new_postflop_with_bets(
        street: Street,
        pot: i32,
        stacks: [i32; 2],
        board: Hand,
        bet_config: Arc<BetConfig>,
    ) -> Self {
        let mut state = Self::new_postflop(street, pot, stacks, board);
        state.bet_config = Some(bet_config);
        state
    }

    /// Classify current node.
    pub fn node_type(&self) -> NodeType {
        // Fold terminal
        if let Some(folder) = self.folded {
            return NodeType::Terminal(TerminalType::Fold(folder));
        }

        // Check allin — both allin means run out board
        if self.is_allin[0] && self.is_allin[1] {
            if self.street == Street::River {
                return NodeType::Terminal(TerminalType::Showdown);
            }
            return NodeType::Chance(self.street.next().unwrap());
        }

        // One allin, other called => run out board
        if (self.is_allin[0] || self.is_allin[1]) && self.closed {
            if self.street == Street::River {
                return NodeType::Terminal(TerminalType::Showdown);
            }
            return NodeType::Chance(self.street.next().unwrap());
        }

        // Action closed (both acted, bets equalized)
        if self.closed {
            if self.street == Street::River {
                return NodeType::Terminal(TerminalType::Showdown);
            }
            return NodeType::Chance(self.street.next().unwrap());
        }

        NodeType::Decision(self.to_act)
    }

    /// Get legal actions at current decision node.
    pub fn actions(&self) -> Vec<Action> {
        let p = self.to_act as usize;
        let opp = 1 - p;
        let my_bet = self.bets[p];
        let opp_bet = self.bets[opp];
        let my_stack = self.stacks[p];
        let to_call = opp_bet - my_bet;

        let max_raises = self.bet_config.as_ref()
            .map_or(DEFAULT_MAX_RAISES, |c| c.max_raises);
        let allin_threshold = self.bet_config.as_ref()
            .map_or(0.0, |c| c.allin_threshold);
        let allin_pot_ratio = self.bet_config.as_ref()
            .map_or(0.0, |c| c.allin_pot_ratio);
        let geometric_2bets = self.bet_config.as_ref()
            .map_or(false, |c| c.geometric_2bets);

        let mut actions = Vec::new();

        if to_call > 0 {
            // Facing a bet/raise
            actions.push(Action::Fold);
            if to_call >= my_stack {
                actions.push(Action::AllIn);
                return actions;
            }
            actions.push(Action::Call);

            // Raises if allowed
            if self.n_raises < max_raises {
                let raise_depth = self.n_raises;
                let current_pot = self.pot + self.bets[0] + self.bets[1];
                let pot_after_call = current_pot + to_call;
                let bets_remaining = max_raises - self.n_raises;
                let use_geometric = geometric_2bets && bets_remaining <= 2;

                let mut any_raise_needs_allin = false;

                if use_geometric {
                    // Geometric sizing: compute the raise that leads to all-in in `bets_remaining` bets
                    let stack_after_call = my_stack - to_call;
                    if stack_after_call > 0 {
                        let geo_raise = geometric_bet_amount(pot_after_call, stack_after_call, bets_remaining as u8);
                        let total_bet = opp_bet + geo_raise;
                        let chips_needed = total_bet - my_bet;
                        if chips_needed < my_stack && geo_raise >= opp_bet - self.min_previous_bet() {
                            // Convert to Frac: geo_raise / pot_after_call as fraction
                            let n = geo_raise as i32;
                            let d = pot_after_call as i32;
                            if d > 0 {
                                actions.push(Action::Bet(BetSize::Frac(n, d)));
                            }
                        } else {
                            any_raise_needs_allin = true;
                        }
                    }
                } else {
                    let sizes = self.get_bet_sizes(raise_depth);
                    for &size in sizes {
                        let raise_amount = compute_bet_amount(size, pot_after_call);
                        let total_bet = opp_bet + raise_amount;
                        let chips_needed = total_bet - my_bet;

                        if chips_needed >= my_stack {
                            any_raise_needs_allin = true;
                            continue;
                        }
                        if raise_amount < opp_bet - self.min_previous_bet() {
                            continue;
                        }
                        // All-in threshold: collapse near-all-in raises to all-in
                        if allin_threshold > 0.0 && my_stack > 0
                            && (chips_needed as f64) >= (my_stack as f64) * (allin_threshold as f64)
                        {
                            any_raise_needs_allin = true;
                            continue;
                        }
                        actions.push(Action::Bet(size));
                    }
                }

                // All-in as a RAISE: available when raises are still allowed.
                // Always allowed if sized raises exceed stack (any_raise_needs_allin).
                // Otherwise subject to allin_pot_ratio.
                if my_stack > to_call {
                    if any_raise_needs_allin
                        || allin_pot_ratio <= 0.0
                        || (my_stack as f64) <= (current_pot as f64) * (allin_pot_ratio as f64)
                    {
                        actions.push(Action::AllIn);
                    }
                }
            }
        } else {
            actions.push(Action::Check);

            // No-donk: OOP can only check when first to act on the FLOP.
            // GTO+ "Donk OFF" disables donk bets on the flop only;
            // OOP can still lead (probe) on turn and river.
            let no_donk = self.bet_config.as_ref().map_or(false, |c| c.no_donk);
            if no_donk && self.to_act == OOP && self.street == Street::Flop {
                return actions;
            }

            let bet_depth = 0u8;
            let current_pot = self.pot + self.bets[0] + self.bets[1];
            // For opening bet: bets_remaining = max_raises (n_raises=0)
            let bets_remaining_bet = max_raises;
            let use_geometric_bet = geometric_2bets && bets_remaining_bet <= 2;

            let mut any_bet_needs_allin = false;
            if use_geometric_bet {
                let geo_amount = geometric_bet_amount(current_pot, my_stack, bets_remaining_bet as u8);
                if geo_amount < my_stack && geo_amount >= 1 {
                    let n = geo_amount as i32;
                    let d = current_pot as i32;
                    if d > 0 {
                        actions.push(Action::Bet(BetSize::Frac(n, d)));
                    }
                } else {
                    any_bet_needs_allin = true;
                }
            } else {
            let sizes = self.get_bet_sizes(bet_depth);
            for &size in sizes {
                let amount = compute_bet_amount(size, current_pot);
                if amount >= my_stack {
                    // Bet exceeds stack — needs AllIn to represent this action
                    any_bet_needs_allin = true;
                    continue;
                }
                if amount < 1 {
                    continue;
                }
                // All-in threshold: collapse near-all-in bets to all-in
                if allin_threshold > 0.0 && my_stack > 0
                    && (amount as f64) >= (my_stack as f64) * (allin_threshold as f64)
                {
                    any_bet_needs_allin = true;
                    continue;
                }
                actions.push(Action::Bet(size));
            }
            } // close else for use_geometric_bet

            if my_stack > 0 {
                if any_bet_needs_allin
                    || allin_pot_ratio <= 0.0
                    || (my_stack as f64) <= (current_pot as f64) * (allin_pot_ratio as f64)
                {
                    actions.push(Action::AllIn);
                }
            }
        }

        actions
    }

    /// Get bet sizes for current street and depth, using custom config if available.
    fn get_bet_sizes(&self, depth: u8) -> &[BetSize] {
        match &self.bet_config {
            Some(config) => config.bet_sizes(self.street, depth),
            None => default_bet_sizes(self.street, depth),
        }
    }

    fn min_previous_bet(&self) -> i32 {
        // Minimum raise = last raise size or 1bb (simplified)
        1
    }

    /// Apply an action, returning new state.
    pub fn apply(&self, action: Action) -> GameState {
        let mut s = self.clone();
        let p = s.to_act as usize;
        let opp = 1 - p;

        match action {
            Action::Fold => {
                s.folded = Some(p as Player);
                return s;
            }
            Action::Check => {
                s.actions_this_street += 1;
                // If both checked (second player checks), close action
                if s.actions_this_street >= 2 || (s.street == Street::Preflop && p == IP as usize && s.bets[0] == s.bets[1]) {
                    s.closed = true;
                } else {
                    s.to_act = opp as Player;
                }
            }
            Action::Call => {
                let to_call = s.bets[opp] - s.bets[p];
                let actual_call = to_call.min(s.stacks[p]);
                s.stacks[p] -= actual_call;
                s.bets[p] += actual_call;
                s.actions_this_street += 1;
                if s.stacks[p] == 0 {
                    s.is_allin[p] = true;
                }
                s.closed = true;
            }
            Action::Bet(size) => {
                let current_pot = s.pot + s.bets[0] + s.bets[1];
                let opp_bet = s.bets[opp];
                let my_bet = s.bets[p];
                let to_call = opp_bet - my_bet;

                let amount = if to_call > 0 {
                    // This is a raise
                    let pot_after_call = current_pot + to_call;
                    let raise_amount = compute_bet_amount(size, pot_after_call);
                    let total_bet = opp_bet + raise_amount;
                    total_bet - my_bet
                } else {
                    // This is a bet
                    compute_bet_amount(size, current_pot)
                };

                let actual = amount.min(s.stacks[p]);
                s.stacks[p] -= actual;
                s.bets[p] += actual;
                s.actions_this_street += 1;
                s.n_raises += 1;
                s.last_aggressor = Some(p as Player);
                if s.stacks[p] == 0 {
                    s.is_allin[p] = true;
                }
                s.to_act = opp as Player;
            }
            Action::AllIn => {
                let allin_amount = s.stacks[p];
                s.bets[p] += allin_amount;
                s.stacks[p] = 0;
                s.is_allin[p] = true;
                s.actions_this_street += 1;

                let to_call = s.bets[opp] - (s.bets[p] - allin_amount);
                if allin_amount <= to_call {
                    // Allin is a call (or less)
                    s.closed = true;
                } else {
                    // Allin is a raise
                    s.n_raises += 1;
                    s.last_aggressor = Some(p as Player);
                    s.to_act = opp as Player;
                }
            }
        }

        s
    }

    /// Deal next street cards (chance transition).
    pub fn deal_street(&self, cards: &[Card]) -> GameState {
        let mut s = self.clone();
        let next_street = self.street.next().expect("no next street");

        // Move bets to pot
        s.pot += s.bets[0] + s.bets[1];
        s.bets = [0, 0];
        s.street = next_street;
        s.n_raises = 0;
        s.actions_this_street = 0;
        s.closed = false;
        s.to_act = OOP; // OOP acts first postflop

        for &c in cards {
            s.board = s.board.add(c);
        }

        // If one or both allin, go straight to chance/terminal
        if s.is_allin[0] || s.is_allin[1] {
            s.closed = true;
        }

        s
    }

    /// Set hole cards.
    pub fn set_holes(&self, p: Player, hole: Hand) -> GameState {
        let mut s = self.clone();
        s.holes[p as usize] = hole;
        s
    }

    /// Terminal payoff for a player (from their perspective).
    /// Positive = profit, negative = loss.
    pub fn payoff(&self, player: Player) -> f32 {
        let p = player as usize;
        let opp = 1 - p;
        let total_pot = self.pot + self.bets[0] + self.bets[1];

        // Fold terminal
        if let Some(folder) = self.folded {
            let invested = self.invested(player);
            if folder == player {
                return -(invested as f32);
            } else {
                return (total_pot as f32) - (invested as f32);
            }
        }

        // Showdown
        let my_hand = self.holes[p].union(self.board);
        let opp_hand = self.holes[opp].union(self.board);

        // Need 5+ cards to evaluate
        if my_hand.count() < 5 || opp_hand.count() < 5 {
            return 0.0; // shouldn't happen in complete games
        }

        let my_strength = evaluate(my_hand);
        let opp_strength = evaluate(opp_hand);

        let invested = self.invested(player);

        if my_strength > opp_strength {
            (total_pot as f32) - (invested as f32)
        } else if my_strength < opp_strength {
            -(invested as f32)
        } else {
            // Tie — split pot
            (total_pot as f32) / 2.0 - (invested as f32)
        }
    }

    /// How much a player has invested total since the game/subgame started.
    pub fn invested(&self, player: Player) -> i32 {
        let p = player as usize;
        self.initial_stacks[p] - self.stacks[p]
    }

}

/// Compute geometric bet size for n bets remaining to reach all-in.
/// Returns the bet amount in chips (first bet of the geometric sequence).
/// Formula: r = ((1 + 2S/P)^(1/n) - 1) / 2, bet = r * P
/// where P = effective pot, S = remaining stack.
fn geometric_bet_amount(pot: i32, stack: i32, bets_remaining: u8) -> i32 {
    if stack <= 0 || pot <= 0 || bets_remaining == 0 {
        return stack.max(0);
    }
    if bets_remaining == 1 {
        return stack; // just all-in
    }
    let p = pot as f64;
    let s = stack as f64;
    let n = bets_remaining as f64;
    let r = ((1.0 + 2.0 * s / p).powf(1.0 / n) - 1.0) / 2.0;
    let amount = (r * p).round() as i32;
    // Clamp: at least 1 chip, at most all-in
    amount.max(1).min(stack)
}

/// Compute bet amount in chips from a BetSize and current pot.
/// Uses banker's rounding (round half to even) to match GTO+ behavior.
fn compute_bet_amount(size: BetSize, pot: i32) -> i32 {
    match size {
        BetSize::Frac(n, d) => {
            let num = pot as i64 * n as i64;
            let d = d as i64;
            // Round to nearest, ties round up (standard rounding)
            ((num + d / 2) / d) as i32
        }
        BetSize::Bb(bb) => bb,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_hand() {
        let g = GameState::new_hand(100);
        assert_eq!(g.street, Street::Preflop);
        assert_eq!(g.bets, [1, 2]);
        assert_eq!(g.stacks, [99, 98]);
        assert_eq!(g.to_act, OOP);
        assert_eq!(g.pot, 0);
    }

    #[test]
    fn test_preflop_actions() {
        let g = GameState::new_hand(100);
        let actions = g.actions();
        // SB facing BB's 2bb: fold, call, raises, allin
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(actions.contains(&Action::AllIn));
        // Should have some raise sizes
        assert!(actions.len() >= 4);
    }

    #[test]
    fn test_fold() {
        let g = GameState::new_hand(100);
        let g2 = g.apply(Action::Fold);
        assert_eq!(g2.folded, Some(OOP));
        assert!(matches!(g2.node_type(), NodeType::Terminal(TerminalType::Fold(OOP))));
        assert_eq!(g2.bets[0], 1);
        assert_eq!(g2.bets[1], 2);
    }

    #[test]
    fn test_call_preflop() {
        let g = GameState::new_hand(100);
        let g2 = g.apply(Action::Call); // SB calls BB
        assert!(g2.closed); // Action closed, go to flop
        assert_eq!(g2.bets, [2, 2]);
        assert_eq!(g2.stacks[0], 98);
    }

    #[test]
    fn test_check_postflop() {
        let g = GameState::new_postflop(Street::Flop, 10, [95, 95], Hand::new());
        assert_eq!(g.to_act, OOP);
        let actions = g.actions();
        assert!(actions.contains(&Action::Check));

        // OOP checks
        let g2 = g.apply(Action::Check);
        assert_eq!(g2.to_act, IP);
        assert!(!g2.closed);

        // IP checks
        let g3 = g2.apply(Action::Check);
        assert!(g3.closed);
    }

    #[test]
    fn test_bet_call_postflop() {
        let g = GameState::new_postflop(Street::River, 20, [90, 90], Hand::new());
        // OOP bets 33% pot (= 20*33/100 = 6.6, rounds to 7)
        let g2 = g.apply(Action::Bet(BetSize::Frac(33, 100)));
        assert_eq!(g2.bets[0], 7); // 20 * 33/100 = 6.6 → 7 (rounded)
        assert_eq!(g2.to_act, IP);

        // IP calls
        let g3 = g2.apply(Action::Call);
        assert!(g3.closed);
        assert_eq!(g3.bets, [7, 7]);
    }

    #[test]
    fn test_street_transition() {
        let g = GameState::new_hand(100);
        let g2 = g.apply(Action::Call); // SB calls BB, preflop closed
        assert!(g2.closed);

        // Deal flop
        use crate::card::card;
        let g3 = g2.deal_street(&[card(12, 3), card(11, 2), card(8, 1)]);
        assert_eq!(g3.street, Street::Flop);
        assert_eq!(g3.pot, 4); // 2+2
        assert_eq!(g3.bets, [0, 0]);
        assert_eq!(g3.to_act, OOP);
        assert_eq!(g3.board.count(), 3);
    }

    #[test]
    fn test_node_type() {
        let g = GameState::new_hand(100);
        assert_eq!(g.node_type(), NodeType::Decision(OOP));

        let g2 = g.apply(Action::Call);
        assert!(matches!(g2.node_type(), NodeType::Chance(Street::Flop)));

        let g3 = g.apply(Action::Fold);
        assert!(matches!(g3.node_type(), NodeType::Terminal(TerminalType::Fold(_))));
    }

    #[test]
    fn test_allin_preflop() {
        let g = GameState::new_hand(100);
        let g2 = g.apply(Action::AllIn); // SB shoves
        assert_eq!(g2.stacks[0], 0);
        assert!(g2.is_allin[0]);
        assert_eq!(g2.to_act, IP);

        let g3 = g2.apply(Action::Call); // BB calls
        assert!(g3.closed);
        // Both allin, should be chance node (need to run out board)
    }

    #[test]
    fn test_invested() {
        let g = GameState::new_hand(100);
        assert_eq!(g.invested(OOP), 1);
        assert_eq!(g.invested(IP), 2);

        let g2 = g.apply(Action::Call);
        assert_eq!(g2.invested(OOP), 2);
        assert_eq!(g2.invested(IP), 2);
    }
}
