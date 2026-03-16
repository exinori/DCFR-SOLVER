/// 6-max Preflop MCCFR Solver
///
/// External Sampling MCCFR with CFR+ and Linear Weighting
/// for computing GTO preflop strategies in 6-max cash games.
///
/// Chip unit: 1 chip = 0.5bb (SB=1, BB=2, stack=200 for 100bb)

use crate::card::{Card, Hand, NUM_CARDS};
use crate::eval::{evaluate, Strength};
use crate::infoset::RegretEntry;
use crate::iso::canonical_hand;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const NUM_PLAYERS: usize = 6;
pub const POSITION_NAMES: [&str; 6] = ["UTG", "HJ", "CO", "BTN", "SB", "BB"];

// ---------------------------------------------------------------------------
// PreflopAction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreflopAction {
    Fold,
    Check,
    Call,
    Raise(i32), // total bet amount in chips
    AllIn,
}

impl std::fmt::Display for PreflopAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreflopAction::Fold => write!(f, "fold"),
            PreflopAction::Check => write!(f, "check"),
            PreflopAction::Call => write!(f, "call"),
            PreflopAction::Raise(total) => write!(f, "raise {}", total),
            PreflopAction::AllIn => write!(f, "allin"),
        }
    }
}

// ---------------------------------------------------------------------------
// PreflopBetConfig
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PreflopBetConfig {
    /// raise_sizes[depth] = list of total bet sizes in chips
    /// depth 0 = open (RFI), depth 1 = 3-bet, depth 2 = 4-bet, ...
    /// Beyond last depth: all-in only
    pub raise_sizes: Vec<Vec<i32>>,
    pub sb_limp: bool,
    pub sb_open_size: Option<i32>,
    /// Minimum n_raises before all-in is allowed as a voluntary action.
    /// 0 = always allowed, 1 = only after someone opens, 2 = after 3-bet, etc.
    /// Does not affect forced all-in (when to_call >= stack).
    pub min_allin_depth: u8,
}

impl Default for PreflopBetConfig {
    fn default() -> Self {
        Self::nl500()
    }
}

impl PreflopBetConfig {
    /// NL500 standard: 2.5bb open, 3.5bb SB, 9bb 3bet, 22bb 4bet
    pub fn nl500() -> Self {
        PreflopBetConfig {
            raise_sizes: vec![
                vec![5],   // depth 0 (open): 2.5bb = 5 chips
                vec![18],  // depth 1 (3-bet): 9bb = 18 chips
                vec![44],  // depth 2 (4-bet): 22bb = 44 chips
                // depth 3+: all-in only
            ],
            sb_limp: true,
            sb_open_size: Some(7), // SB open: 3.5bb = 7 chips
            min_allin_depth: 1,    // No open-shove; all-in allowed after first raise
        }
    }
}

// ---------------------------------------------------------------------------
// PreflopState
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PreflopState {
    pub stacks: [i32; NUM_PLAYERS],
    pub bets: [i32; NUM_PLAYERS],
    pub folded: [bool; NUM_PLAYERS],
    pub has_acted: [bool; NUM_PLAYERS],
    pub all_in: [bool; NUM_PLAYERS],
    pub to_act: u8,
    pub n_raises: u8,
    pub holes: [Hand; NUM_PLAYERS],
    pub config: PreflopBetConfig,
    pub last_raise_size: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflopNodeType {
    TerminalFold(u8),   // winner position
    TerminalShowdown,
    Decision(u8),       // player position
}

impl PreflopState {
    pub fn new_6max(config: PreflopBetConfig) -> Self {
        let mut stacks = [200; NUM_PLAYERS]; // 100bb = 200 chips
        let mut bets = [0i32; NUM_PLAYERS];
        stacks[4] -= 1; // SB posts 1 chip (0.5bb)
        stacks[5] -= 2; // BB posts 2 chips (1bb)
        bets[4] = 1;
        bets[5] = 2;

        PreflopState {
            stacks,
            bets,
            folded: [false; NUM_PLAYERS],
            has_acted: [false; NUM_PLAYERS],
            all_in: [false; NUM_PLAYERS],
            to_act: 0, // UTG acts first
            n_raises: 0, // Blinds are not a "raise"; open = depth 0
            holes: [Hand::new(); NUM_PLAYERS],
            config,
            last_raise_size: 2, // BB = 2 chips as min raise reference
        }
    }

    /// Heads-up (SB vs BB only). Positions 0-3 pre-folded.
    pub fn new_heads_up(config: PreflopBetConfig) -> Self {
        let mut stacks = [0i32; NUM_PLAYERS];
        let mut bets = [0i32; NUM_PLAYERS];
        stacks[4] = 200 - 1;
        stacks[5] = 200 - 2;
        bets[4] = 1;
        bets[5] = 2;

        PreflopState {
            stacks,
            bets,
            folded: [true, true, true, true, false, false],
            has_acted: [true, true, true, true, false, false],
            all_in: [false; NUM_PLAYERS],
            to_act: 4, // SB acts first
            n_raises: 0,
            holes: [Hand::new(); NUM_PLAYERS],
            config,
            last_raise_size: 2,
        }
    }

    pub fn active_count(&self) -> usize {
        (0..NUM_PLAYERS).filter(|&i| !self.folded[i]).count()
    }

    fn active_non_allin_count(&self) -> usize {
        (0..NUM_PLAYERS).filter(|&i| !self.folded[i] && !self.all_in[i]).count()
    }

    pub fn max_bet(&self) -> i32 {
        self.bets.iter().copied().max().unwrap()
    }

    pub fn is_closed(&self) -> bool {
        if self.active_count() <= 1 {
            return true;
        }
        // If only 1 or 0 non-allin players remain, action is closed
        if self.active_non_allin_count() <= 1 {
            // But only if all active non-allin players have acted and matched
            let mb = self.max_bet();
            for i in 0..NUM_PLAYERS {
                if self.folded[i] || self.all_in[i] {
                    continue;
                }
                if !self.has_acted[i] || self.bets[i] != mb {
                    return false;
                }
            }
            return true;
        }
        // All active, non-allin players must have acted and matched max bet
        let mb = self.max_bet();
        for i in 0..NUM_PLAYERS {
            if self.folded[i] || self.all_in[i] {
                continue;
            }
            if !self.has_acted[i] || self.bets[i] != mb {
                return false;
            }
        }
        true
    }

    pub fn node_type(&self) -> PreflopNodeType {
        // Check fold terminal: exactly 1 player left
        if self.active_count() == 1 {
            let winner = (0..NUM_PLAYERS).find(|&i| !self.folded[i]).unwrap();
            return PreflopNodeType::TerminalFold(winner as u8);
        }
        if self.is_closed() {
            return PreflopNodeType::TerminalShowdown;
        }
        PreflopNodeType::Decision(self.to_act)
    }

    fn next_active_player(&self, after: u8) -> u8 {
        let mut p = (after as usize + 1) % NUM_PLAYERS;
        for _ in 0..NUM_PLAYERS {
            if !self.folded[p] && !self.all_in[p] {
                return p as u8;
            }
            p = (p + 1) % NUM_PLAYERS;
        }
        after // shouldn't reach here
    }

    pub fn actions(&self) -> Vec<PreflopAction> {
        let p = self.to_act as usize;
        let my_bet = self.bets[p];
        let my_stack = self.stacks[p];
        let max_bet = self.max_bet();
        let to_call = max_bet - my_bet;
        let mut actions = Vec::new();

        if to_call > 0 {
            // Facing a bet/raise
            actions.push(PreflopAction::Fold);

            if to_call >= my_stack {
                // Can only fold or all-in
                actions.push(PreflopAction::AllIn);
                return actions;
            }
            // Open-limp (n_raises==0): only SB can limp; others must fold or raise.
            // Facing a raise (n_raises>0): Call is always available.
            if self.n_raises > 0 || (p == 4 && self.config.sb_limp) {
                actions.push(PreflopAction::Call);
            }

            // Raise options
            let depth = self.n_raises as usize;
            if depth < self.config.raise_sizes.len() {
                let sizes = &self.config.raise_sizes[depth];
                for &total_bet in sizes {
                    // Adjust for SB open
                    let total_bet = if depth == 0 && p == 4 {
                        self.config.sb_open_size.unwrap_or(total_bet)
                    } else {
                        total_bet
                    };
                    // Min raise: raise_amount must be >= last_raise_size
                    let raise_amount = total_bet - max_bet;
                    if raise_amount < self.last_raise_size {
                        continue;
                    }
                    let chips_needed = total_bet - my_bet;
                    if chips_needed >= my_stack {
                        continue;
                    }
                    if chips_needed <= to_call {
                        continue; // not actually a raise
                    }
                    actions.push(PreflopAction::Raise(total_bet));
                }
            }

            // All-in (as a voluntary raise) — only if min_allin_depth met
            if my_stack > to_call && self.n_raises >= self.config.min_allin_depth {
                actions.push(PreflopAction::AllIn);
            }
        } else {
            // BB option (to_call == 0, BB hasn't acted yet) or similar
            actions.push(PreflopAction::Check);

            // Raise options (treat as opening from current depth)
            let depth = self.n_raises as usize;
            if depth < self.config.raise_sizes.len() {
                let sizes = &self.config.raise_sizes[depth];
                for &total_bet in sizes {
                    let chips_needed = total_bet - my_bet;
                    if chips_needed >= my_stack {
                        continue;
                    }
                    if chips_needed <= 0 {
                        continue;
                    }
                    actions.push(PreflopAction::Raise(total_bet));
                }
            }

            if my_stack > 0 && self.n_raises >= self.config.min_allin_depth {
                actions.push(PreflopAction::AllIn);
            }
        }

        actions
    }

    pub fn apply(&self, action: PreflopAction) -> PreflopState {
        let mut s = self.clone();
        let p = s.to_act as usize;

        match action {
            PreflopAction::Fold => {
                s.folded[p] = true;
                s.has_acted[p] = true;
                s.to_act = s.next_active_player(s.to_act);
            }
            PreflopAction::Check => {
                s.has_acted[p] = true;
                s.to_act = s.next_active_player(s.to_act);
            }
            PreflopAction::Call => {
                let max_bet = s.max_bet();
                let to_call = max_bet - s.bets[p];
                let actual = to_call.min(s.stacks[p]);
                s.stacks[p] -= actual;
                s.bets[p] += actual;
                s.has_acted[p] = true;
                if s.stacks[p] == 0 {
                    s.all_in[p] = true;
                }
                s.to_act = s.next_active_player(s.to_act);
            }
            PreflopAction::Raise(total_bet) => {
                let raise_size = total_bet - s.max_bet();
                let chips_needed = total_bet - s.bets[p];
                s.stacks[p] -= chips_needed;
                s.bets[p] = total_bet;
                s.has_acted[p] = true;
                s.n_raises += 1;
                s.last_raise_size = raise_size;
                if s.stacks[p] == 0 {
                    s.all_in[p] = true;
                }
                // Reset has_acted for all other active, non-allin players
                for i in 0..NUM_PLAYERS {
                    if i != p && !s.folded[i] && !s.all_in[i] {
                        s.has_acted[i] = false;
                    }
                }
                s.to_act = s.next_active_player(s.to_act);
            }
            PreflopAction::AllIn => {
                let allin_amount = s.stacks[p];
                let old_max = s.max_bet();
                s.bets[p] += allin_amount;
                s.stacks[p] = 0;
                s.all_in[p] = true;
                s.has_acted[p] = true;

                // If this all-in is a raise (new bet > old max)
                if s.bets[p] > old_max {
                    let raise_size = s.bets[p] - old_max;
                    // Only count as a proper raise if raise_size >= last_raise_size
                    if raise_size >= s.last_raise_size {
                        s.n_raises += 1;
                        s.last_raise_size = raise_size;
                        // Reset has_acted for others
                        for i in 0..NUM_PLAYERS {
                            if i != p && !s.folded[i] && !s.all_in[i] {
                                s.has_acted[i] = false;
                            }
                        }
                    }
                    // Even if not a full raise, opponents who haven't matched need to act
                    for i in 0..NUM_PLAYERS {
                        if i != p && !s.folded[i] && !s.all_in[i] && s.bets[i] < s.bets[p] {
                            s.has_acted[i] = false;
                        }
                    }
                }

                s.to_act = s.next_active_player(s.to_act);
            }
        }

        s
    }

    /// Fold terminal payoff: winner takes all bets
    pub fn payoff_fold(&self, winner: u8) -> [f32; NUM_PLAYERS] {
        let mut payoffs = [0.0f32; NUM_PLAYERS];
        let total_pot: i32 = self.bets.iter().sum();
        for i in 0..NUM_PLAYERS {
            let invested = self.bets[i];
            if i == winner as usize {
                payoffs[i] = (total_pot - invested) as f32;
            } else {
                payoffs[i] = -(invested as f32);
            }
        }
        payoffs
    }

    /// Showdown payoff with side pots.
    /// board: 5-card Hand for evaluation.
    pub fn payoff_showdown(&self, board: Hand) -> [f32; NUM_PLAYERS] {
        let mut payoffs = [0.0f32; NUM_PLAYERS];

        // Collect active players and their hands
        let mut active: Vec<(usize, Strength)> = Vec::new();
        for i in 0..NUM_PLAYERS {
            if self.folded[i] {
                continue;
            }
            let full_hand = self.holes[i].union(board);
            let strength = evaluate(full_hand);
            active.push((i, strength));
        }

        if active.is_empty() {
            return payoffs;
        }

        // Side pot calculation
        // Collect unique bet levels from active players, sorted ascending
        let mut bet_levels: Vec<i32> = active.iter().map(|&(i, _)| self.bets[i]).collect();
        bet_levels.sort();
        bet_levels.dedup();

        let mut prev_level = 0i32;
        for &level in &bet_levels {
            if level <= prev_level {
                continue;
            }
            // This pot layer: each player contributes min(their bet, level) - min(their bet, prev_level)
            let mut pot_layer = 0i32;
            let mut eligible: Vec<(usize, Strength)> = Vec::new();

            for i in 0..NUM_PLAYERS {
                let contrib = self.bets[i].min(level) - self.bets[i].min(prev_level);
                pot_layer += contrib;
                // Eligible = active (not folded) and bet >= level
                if !self.folded[i] && self.bets[i] >= level {
                    if let Some(&(_, str)) = active.iter().find(|&&(idx, _)| idx == i) {
                        eligible.push((i, str));
                    }
                }
            }

            if pot_layer > 0 && !eligible.is_empty() {
                // Find best hand among eligible
                let best = eligible.iter().map(|&(_, s)| s).max().unwrap();
                let winners: Vec<usize> = eligible.iter()
                    .filter(|&&(_, s)| s == best)
                    .map(|&(i, _)| i)
                    .collect();
                let share = pot_layer as f32 / winners.len() as f32;
                for &w in &winners {
                    payoffs[w] += share;
                }
            }

            prev_level = level;
        }

        // Convert to profit/loss (subtract invested)
        for i in 0..NUM_PLAYERS {
            payoffs[i] -= self.bets[i] as f32;
        }

        payoffs
    }
}

// ---------------------------------------------------------------------------
// PreflopInfoKey
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PreflopInfoKey {
    pub bucket: u8,       // canonical hand index 0..168
    pub history: Vec<u8>, // action indices from root
}

// ---------------------------------------------------------------------------
// PreflopBlueprint
// ---------------------------------------------------------------------------

pub struct PreflopBlueprint {
    pub entries: HashMap<PreflopInfoKey, RegretEntry>,
    pub iterations: u64,
    pub config: PreflopBetConfig,
}

impl PreflopBlueprint {
    pub fn new(config: PreflopBetConfig) -> Self {
        PreflopBlueprint {
            entries: HashMap::new(),
            iterations: 0,
            config,
        }
    }

    fn get_or_create(&mut self, key: &PreflopInfoKey, n_actions: usize) -> &mut RegretEntry {
        if !self.entries.contains_key(key) {
            self.entries.insert(key.clone(), RegretEntry::new(n_actions));
        }
        self.entries.get_mut(key).unwrap()
    }

    pub fn save<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        // Config
        let n_depths = self.config.raise_sizes.len() as u8;
        w.write_u8(n_depths)?;
        for sizes in &self.config.raise_sizes {
            w.write_u8(sizes.len() as u8)?;
            for &s in sizes {
                w.write_i32::<LittleEndian>(s)?;
            }
        }
        w.write_u8(self.config.sb_limp as u8)?;
        w.write_i32::<LittleEndian>(self.config.sb_open_size.unwrap_or(-1))?;
        w.write_u8(self.config.min_allin_depth)?;

        // Entries
        w.write_u64::<LittleEndian>(self.iterations)?;
        w.write_u64::<LittleEndian>(self.entries.len() as u64)?;

        for (key, entry) in &self.entries {
            w.write_u8(key.bucket)?;
            w.write_u16::<LittleEndian>(key.history.len() as u16)?;
            for &h in &key.history {
                w.write_u8(h)?;
            }
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

        // Config
        let n_depths = r.read_u8()? as usize;
        let mut raise_sizes = Vec::with_capacity(n_depths);
        for _ in 0..n_depths {
            let n = r.read_u8()? as usize;
            let mut sizes = Vec::with_capacity(n);
            for _ in 0..n {
                sizes.push(r.read_i32::<LittleEndian>()?);
            }
            raise_sizes.push(sizes);
        }
        let sb_limp = r.read_u8()? != 0;
        let sb_open_raw = r.read_i32::<LittleEndian>()?;
        let sb_open_size = if sb_open_raw < 0 { None } else { Some(sb_open_raw) };
        let min_allin_depth = r.read_u8()?;

        let config = PreflopBetConfig { raise_sizes, sb_limp, sb_open_size, min_allin_depth };

        // Entries
        let iterations = r.read_u64::<LittleEndian>()?;
        let n_entries = r.read_u64::<LittleEndian>()? as usize;

        let mut entries = HashMap::with_capacity(n_entries);
        for _ in 0..n_entries {
            let bucket = r.read_u8()?;
            let hist_len = r.read_u16::<LittleEndian>()? as usize;
            let mut history = Vec::with_capacity(hist_len);
            for _ in 0..hist_len {
                history.push(r.read_u8()?);
            }
            let key = PreflopInfoKey { bucket, history };

            let n_actions = r.read_u16::<LittleEndian>()? as usize;
            let mut regrets = vec![0.0f32; n_actions];
            for i in 0..n_actions {
                regrets[i] = r.read_f32::<LittleEndian>()?;
            }
            let mut cum_strategy = vec![0.0f32; n_actions];
            for i in 0..n_actions {
                cum_strategy[i] = r.read_f32::<LittleEndian>()?;
            }
            entries.insert(key, RegretEntry { regrets, cum_strategy, n_actions });
        }

        Ok(PreflopBlueprint { entries, iterations, config })
    }
}

// ---------------------------------------------------------------------------
// PreflopTrainer
// ---------------------------------------------------------------------------

pub struct PreflopTrainer {
    pub blueprint: PreflopBlueprint,
    rng: SmallRng,
    /// Number of board samples to average at showdown terminals.
    /// Higher = less variance, slower per iteration.
    pub board_samples: usize,
    /// OOP position tax as fraction of total pot.
    /// Transfers pot_size * oop_pot_tax from OOP to IP at showdown.
    /// Models the fact that being out of position realizes less equity.
    /// Default 0.20: OOP loses 20% of pot (scaled by position gap), IP gains same.
    /// Set to 0.0 to disable (pure raw equity, like v1).
    pub oop_pot_tax: f32,
}

impl PreflopTrainer {
    pub fn new(config: PreflopBetConfig, seed: u64) -> Self {
        PreflopTrainer {
            blueprint: PreflopBlueprint::new(config),
            rng: SmallRng::seed_from_u64(seed),
            board_samples: 10,
            oop_pot_tax: 0.20,
        }
    }

    pub fn train(&mut self, iterations: u64) {
        for i in 0..iterations {
            if i > 0 && i % 100_000 == 0 {
                eprintln!(
                    "  iteration {}/{} ({} info sets)",
                    i, iterations, self.blueprint.entries.len()
                );
            }

            // Deal random hole cards to 6 players (12 cards, no conflicts)
            let holes = self.deal_holes();

            // Traverser cycles through 6 positions
            let traverser = (i % NUM_PLAYERS as u64) as u8;

            let mut state = PreflopState::new_6max(self.blueprint.config.clone());
            for p in 0..NUM_PLAYERS {
                state.holes[p] = holes[p];
            }

            let mut history = Vec::new();
            self.cfr_external(&state, traverser, &mut history);
            self.blueprint.iterations += 1;
        }
    }

    /// Heads-up training (SB vs BB only). Much faster convergence.
    pub fn train_hu(&mut self, iterations: u64) {
        for i in 0..iterations {
            if i > 0 && i % 100_000 == 0 {
                eprintln!(
                    "  hu iteration {}/{} ({} info sets)",
                    i, iterations, self.blueprint.entries.len()
                );
            }

            // Deal only to SB(4) and BB(5)
            let mut dead = Hand::new();
            let c1 = self.draw_excluding(dead); dead = dead.add(c1);
            let c2 = self.draw_excluding(dead); dead = dead.add(c2);
            let c3 = self.draw_excluding(dead); dead = dead.add(c3);
            let c4 = self.draw_excluding(dead); dead = dead.add(c4);

            let mut state = PreflopState::new_heads_up(self.blueprint.config.clone());
            state.holes[4] = Hand::new().add(c1).add(c2);
            state.holes[5] = Hand::new().add(c3).add(c4);

            // Traverser alternates SB(4) and BB(5)
            let traverser = if i % 2 == 0 { 4 } else { 5 };

            let mut history = Vec::new();
            self.cfr_external(&state, traverser, &mut history);
            self.blueprint.iterations += 1;
        }
    }

    fn deal_holes(&mut self) -> [Hand; NUM_PLAYERS] {
        let mut holes = [Hand::new(); NUM_PLAYERS];
        let mut dead = Hand::new();
        for p in 0..NUM_PLAYERS {
            let c1 = self.draw_excluding(dead);
            dead = dead.add(c1);
            let c2 = self.draw_excluding(dead);
            dead = dead.add(c2);
            holes[p] = Hand::new().add(c1).add(c2);
        }
        holes
    }

    fn draw_excluding(&mut self, dead: Hand) -> Card {
        loop {
            let c = self.rng.gen_range(0..NUM_CARDS as u8);
            if !dead.contains(c) {
                return c;
            }
        }
    }

    fn cfr_external(
        &mut self,
        state: &PreflopState,
        traverser: u8,
        history: &mut Vec<u8>,
    ) -> f32 {
        match state.node_type() {
            PreflopNodeType::TerminalFold(winner) => {
                let payoffs = state.payoff_fold(winner);
                payoffs[traverser as usize]
            }
            PreflopNodeType::TerminalShowdown => {
                // Average payoff over multiple board samples to reduce variance
                let mut dead = Hand::new();
                for p in 0..NUM_PLAYERS {
                    dead = dead.union(state.holes[p]);
                }
                let n = self.board_samples;
                let mut total = 0.0f32;

                // Determine IP/OOP for position tax adjustment.
                // Among active players, IP = closest to BTN (position index 3).
                // Position order: UTG(0), HJ(1), CO(2), BTN(3), SB(4), BB(5)
                // Postflop order: SB/BB act first (OOP), then UTG→BTN (IP).
                let base_tax = self.oop_pot_tax;
                let need_adjust = base_tax > 0.001 && state.active_count() == 2;
                let (ip_idx, oop_idx, tax_amount) = if need_adjust {
                    let active: Vec<usize> = (0..NUM_PLAYERS)
                        .filter(|&i| !state.folded[i])
                        .collect();
                    debug_assert_eq!(active.len(), 2);
                    // Postflop action order: SB → BB → UTG → HJ → CO → BTN
                    // Higher rank = acts later = more IP
                    let postflop_rank = |p: usize| -> usize {
                        match p {
                            3 => 5, // BTN (always IP)
                            2 => 4, // CO
                            1 => 3, // HJ
                            0 => 2, // UTG
                            5 => 1, // BB (IP vs SB only)
                            4 => 0, // SB (always OOP)
                            _ => 0,
                        }
                    };
                    let (ip, oop) = if postflop_rank(active[0]) > postflop_rank(active[1]) {
                        (active[0], active[1])
                    } else {
                        (active[1], active[0])
                    };
                    // Scale tax by positional gap (max gap = 5, BTN vs SB)
                    let gap = postflop_rank(ip) - postflop_rank(oop);
                    let scaled_tax = base_tax * gap as f32 / 5.0;
                    let total_pot: i32 = state.bets.iter().sum();
                    let tax = total_pot as f32 * scaled_tax;
                    (ip, oop, tax)
                } else {
                    (0, 0, 0.0)
                };

                for _ in 0..n {
                    let board = self.sample_board(dead);
                    let mut payoffs = state.payoff_showdown(board);

                    // Position tax: OOP pays a fixed fraction of pot to IP
                    if need_adjust {
                        payoffs[oop_idx] -= tax_amount;
                        payoffs[ip_idx] += tax_amount;
                    }

                    total += payoffs[traverser as usize];
                }
                total / n as f32
            }
            PreflopNodeType::Decision(player) => {
                let actions = state.actions();
                let n_actions = actions.len();

                // Get canonical hand bucket for current player
                let hole = state.holes[player as usize];
                let cards: Vec<Card> = hole.iter().collect();
                let bucket = canonical_hand(cards[0], cards[1]).index();

                let key = PreflopInfoKey {
                    bucket,
                    history: history.clone(),
                };

                let strategy = {
                    let entry = self.blueprint.get_or_create(&key, n_actions);
                    entry.current_strategy()
                };

                if player == traverser {
                    // Explore all actions
                    let mut action_values = vec![0.0f32; n_actions];
                    let mut node_value = 0.0f32;

                    for (a, &action) in actions.iter().enumerate() {
                        let child = state.apply(action);
                        history.push(a as u8);
                        let value = self.cfr_external(&child, traverser, history);
                        history.pop();
                        action_values[a] = value;
                        node_value += strategy[a] * value;
                    }

                    // Update regrets (CFR+: floor at 0) + cumulative strategy
                    let weight = self.blueprint.iterations as f32 + 1.0;
                    let entry = self.blueprint.get_or_create(&key, n_actions);
                    for a in 0..n_actions {
                        entry.regrets[a] = (entry.regrets[a] + action_values[a] - node_value).max(0.0);
                        entry.cum_strategy[a] += weight * strategy[a];
                    }

                    node_value
                } else {
                    // Opponent: sample one action
                    let action_idx = sample_action(&strategy, &mut self.rng);
                    let child = state.apply(actions[action_idx]);
                    history.push(action_idx as u8);
                    let value = self.cfr_external(&child, traverser, history);
                    history.pop();
                    value
                }
            }
        }
    }

    fn sample_board(&mut self, dead: Hand) -> Hand {
        let mut board = Hand::new();
        let mut dead = dead;
        for _ in 0..5 {
            let c = self.draw_excluding(dead);
            board = board.add(c);
            dead = dead.add(c);
        }
        board
    }
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
// Chart Extraction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct PreflopSpot {
    pub spot_name: String,
    pub hands: Vec<HandStrategy>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HandStrategy {
    pub hand: String,
    pub actions: Vec<ActionProb>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ActionProb {
    pub action: String,
    pub prob: f32,
}

impl PreflopBlueprint {
    /// Extract opening (RFI) charts for each position.
    pub fn extract_charts(&self) -> Vec<PreflopSpot> {
        let config = &self.config;
        let mut spots = Vec::new();

        // RFI spots: UTG through BTN (positions 0-3)
        // These players face the initial state (no one has opened)
        // History encodes folds before them
        for pos in 0..4 {
            let spot_name = format!("{} RFI", POSITION_NAMES[pos]);
            let history_prefix: Vec<u8> = (0..pos).map(|_| 0u8).collect(); // all fold = action index 0

            let mut hands = Vec::new();
            for idx in 0..169u8 {
                let ch = crate::iso::CanonicalHand::from_index(idx);
                let key = PreflopInfoKey {
                    bucket: idx,
                    history: history_prefix.clone(),
                };
                if let Some(entry) = self.entries.get(&key) {
                    let avg = entry.average_strategy();
                    // Build action labels from the state at this point
                    let state = build_state_from_folds(pos, config);
                    let actions = state.actions();
                    let action_probs: Vec<ActionProb> = actions.iter()
                        .zip(avg.iter())
                        .map(|(a, &p)| ActionProb {
                            action: format!("{}", a),
                            prob: p,
                        })
                        .collect();
                    hands.push(HandStrategy {
                        hand: ch.to_string(),
                        actions: action_probs,
                    });
                } else {
                    hands.push(HandStrategy {
                        hand: ch.to_string(),
                        actions: vec![],
                    });
                }
            }
            spots.push(PreflopSpot { spot_name, hands });
        }

        // SB RFI (position 4, everyone folds to SB)
        {
            let history_prefix: Vec<u8> = vec![0, 0, 0, 0]; // UTG-BTN all fold
            let mut hands = Vec::new();
            for idx in 0..169u8 {
                let ch = crate::iso::CanonicalHand::from_index(idx);
                let key = PreflopInfoKey {
                    bucket: idx,
                    history: history_prefix.clone(),
                };
                if let Some(entry) = self.entries.get(&key) {
                    let avg = entry.average_strategy();
                    let state = build_state_from_folds(4, config);
                    let actions = state.actions();
                    let action_probs: Vec<ActionProb> = actions.iter()
                        .zip(avg.iter())
                        .map(|(a, &p)| ActionProb { action: format!("{}", a), prob: p })
                        .collect();
                    hands.push(HandStrategy { hand: ch.to_string(), actions: action_probs });
                } else {
                    hands.push(HandStrategy { hand: ch.to_string(), actions: vec![] });
                }
            }
            spots.push(PreflopSpot { spot_name: "SB RFI".to_string(), hands });
        }

        spots
    }
}

/// Build a PreflopState after `pos` players have folded.
fn build_state_from_folds(pos: usize, config: &PreflopBetConfig) -> PreflopState {
    let mut state = PreflopState::new_6max(config.clone());
    for _ in 0..pos {
        state = state.apply(PreflopAction::Fold);
    }
    state
}

// ---------------------------------------------------------------------------
// SRP Matchup Extraction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SrpMatchup {
    pub matchup: String,
    pub pot_chips: i32,
    pub eff_stack_chips: i32,
    pub opener: MatchupSide,
    pub caller: MatchupSide,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MatchupSide {
    pub position: String,
    pub range: std::collections::BTreeMap<String, f32>,
}

impl PreflopBlueprint {
    // -----------------------------------------------------------------------
    // Range extraction helpers
    // -----------------------------------------------------------------------

    /// Extract range from a single decision point: P(action) per canonical hand.
    fn extract_range_single(
        &self,
        history: &[u8],
        action_idx: usize,
    ) -> std::collections::BTreeMap<String, f32> {
        let mut range = std::collections::BTreeMap::new();
        for idx in 0..169u8 {
            let ch = crate::iso::CanonicalHand::from_index(idx);
            let key = PreflopInfoKey { bucket: idx, history: history.to_vec() };
            if let Some(entry) = self.entries.get(&key) {
                let avg = entry.average_strategy();
                let prob = avg[action_idx];
                if prob > 0.001 {
                    range.insert(ch.to_string(), prob);
                }
            }
        }
        range
    }

    /// Extract range from two sequential decision points: P(a1) × P(a2).
    /// Used for compound ranges like open-then-call-3bet.
    fn extract_range_compound(
        &self,
        history_1: &[u8], action_idx_1: usize,
        history_2: &[u8], action_idx_2: usize,
    ) -> std::collections::BTreeMap<String, f32> {
        let mut range = std::collections::BTreeMap::new();
        for idx in 0..169u8 {
            let ch = crate::iso::CanonicalHand::from_index(idx);
            let key1 = PreflopInfoKey { bucket: idx, history: history_1.to_vec() };
            let key2 = PreflopInfoKey { bucket: idx, history: history_2.to_vec() };
            let prob1 = self.entries.get(&key1)
                .map(|e| e.average_strategy()[action_idx_1])
                .unwrap_or(0.0);
            let prob2 = self.entries.get(&key2)
                .map(|e| e.average_strategy()[action_idx_2])
                .unwrap_or(0.0);
            let prob = prob1 * prob2;
            if prob > 0.001 {
                range.insert(ch.to_string(), prob);
            }
        }
        range
    }

    // -----------------------------------------------------------------------
    // SRP Matchup Extraction (15 matchups)
    // -----------------------------------------------------------------------

    /// Extract 15 SRP (Single Raised Pot) matchup ranges.
    /// Each matchup: opener raises, everyone between folds, caller calls.
    pub fn extract_matchups(&self) -> Vec<SrpMatchup> {
        let mut matchups = Vec::new();
        for opener_pos in 0..5 {
            for caller_pos in (opener_pos + 1)..6 {
                matchups.push(self.extract_one_matchup(opener_pos, caller_pos));
            }
        }
        matchups
    }

    fn extract_one_matchup(&self, opener_pos: usize, caller_pos: usize) -> SrpMatchup {
        let config = &self.config;
        let mut state = PreflopState::new_6max(config.clone());
        let mut history: Vec<u8> = Vec::new();

        // Fold everyone before opener
        for _ in 0..opener_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // Opener's history (before their raise)
        let opener_history = history.clone();
        let opener_actions = state.actions();
        let raise_idx = opener_actions.iter()
            .position(|a| matches!(a, PreflopAction::Raise(_)))
            .expect("opener must have a raise action");
        let raise_action = opener_actions[raise_idx];

        // Apply opener's raise
        history.push(raise_idx as u8);
        state = state.apply(raise_action);

        // Fold everyone between opener and caller
        for _ in (opener_pos + 1)..caller_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // Caller's history (before their action)
        let caller_history = history;
        let caller_actions = state.actions();
        let call_idx = caller_actions.iter()
            .position(|a| matches!(a, PreflopAction::Call))
            .expect("caller must have a call action");

        // Extract ranges using helpers
        let opener_range = self.extract_range_single(&opener_history, raise_idx);
        let caller_range = self.extract_range_single(&caller_history, call_idx);

        // Apply call to get pot/stack
        state = state.apply(PreflopAction::Call);
        let pot_chips: i32 = state.bets.iter().sum();
        let eff_stack_chips = state.stacks[opener_pos].min(state.stacks[caller_pos]);

        SrpMatchup {
            matchup: format!("{} vs {}", POSITION_NAMES[opener_pos], POSITION_NAMES[caller_pos]),
            pot_chips,
            eff_stack_chips,
            opener: MatchupSide {
                position: POSITION_NAMES[opener_pos].to_string(),
                range: opener_range,
            },
            caller: MatchupSide {
                position: POSITION_NAMES[caller_pos].to_string(),
                range: caller_range,
            },
        }
    }

    // -----------------------------------------------------------------------
    // 3bet Matchup Extraction (10 matchups)
    // -----------------------------------------------------------------------

    /// Extract 10 specified 3bet pot matchup ranges.
    pub fn extract_3bet_matchups(&self) -> Vec<SrpMatchup> {
        let matchup_list: [(usize, usize); 10] = [
            (0, 3), // UTG vs BTN
            (0, 5), // UTG vs BB
            (1, 2), // HJ vs CO
            (1, 3), // HJ vs BTN
            (1, 5), // HJ vs BB
            (2, 3), // CO vs BTN
            (2, 5), // CO vs BB
            (3, 4), // BTN vs SB
            (3, 5), // BTN vs BB
            (4, 5), // SB vs BB
        ];
        matchup_list.iter()
            .map(|&(opener, bettor3)| self.extract_3bet_matchup(opener, bettor3))
            .collect()
    }

    fn extract_3bet_matchup(&self, opener_pos: usize, bettor3_pos: usize) -> SrpMatchup {
        let config = &self.config;
        let mut state = PreflopState::new_6max(config.clone());
        let mut history: Vec<u8> = Vec::new();

        // 1. Fold everyone before opener
        for _ in 0..opener_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // 2. Opener's decision point — find raise action
        let opener_open_history = history.clone();
        let opener_actions = state.actions();
        let open_raise_idx = opener_actions.iter()
            .position(|a| matches!(a, PreflopAction::Raise(_)))
            .expect("opener must have a raise action");

        // 3. Apply opener's raise
        history.push(open_raise_idx as u8);
        state = state.apply(opener_actions[open_raise_idx]);

        // 4. Fold everyone between opener and 3bettor
        for _ in (opener_pos + 1)..bettor3_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // 5. 3bettor's decision point — find raise action
        let bettor3_history = history.clone();
        let bettor3_actions = state.actions();
        let bet3_raise_idx = bettor3_actions.iter()
            .position(|a| matches!(a, PreflopAction::Raise(_)))
            .expect("3bettor must have a raise action");

        // 6. Apply 3bet
        history.push(bet3_raise_idx as u8);
        state = state.apply(bettor3_actions[bet3_raise_idx]);

        // 7. Fold remaining players until opener gets to act again
        //    state.to_act cycles through active non-folded players
        while state.to_act as usize != opener_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // 8. Opener facing 3bet — find call action
        let opener_call_history = history.clone();
        let opener_facing_actions = state.actions();
        let call_idx = opener_facing_actions.iter()
            .position(|a| matches!(a, PreflopAction::Call))
            .expect("opener must have a call action facing 3bet");

        // 9. Apply call → get pot/stack
        state = state.apply(PreflopAction::Call);
        let pot_chips: i32 = state.bets.iter().sum();
        let eff_stack_chips = state.stacks[opener_pos].min(state.stacks[bettor3_pos]);

        // 10. Extract ranges
        // Opener range = P(open) × P(call 3bet)
        let opener_range = self.extract_range_compound(
            &opener_open_history, open_raise_idx,
            &opener_call_history, call_idx,
        );
        // 3bettor range = P(3bet | facing open)
        let bettor3_range = self.extract_range_single(&bettor3_history, bet3_raise_idx);

        SrpMatchup {
            matchup: format!("{} open vs {} 3bet", POSITION_NAMES[opener_pos], POSITION_NAMES[bettor3_pos]),
            pot_chips,
            eff_stack_chips,
            opener: MatchupSide {
                position: POSITION_NAMES[opener_pos].to_string(),
                range: opener_range,
            },
            caller: MatchupSide {
                position: POSITION_NAMES[bettor3_pos].to_string(),
                range: bettor3_range,
            },
        }
    }

    // -----------------------------------------------------------------------
    // 4bet Matchup Extraction (5 matchups)
    // -----------------------------------------------------------------------

    /// Extract 5 specified 4bet pot matchup ranges.
    pub fn extract_4bet_matchups(&self) -> Vec<SrpMatchup> {
        let matchup_list: [(usize, usize); 5] = [
            (2, 3), // CO vs BTN
            (3, 5), // BTN vs BB
            (1, 3), // HJ vs BTN
            (0, 5), // UTG vs BB
            (4, 5), // SB vs BB
        ];
        matchup_list.iter()
            .map(|&(opener, bettor3)| self.extract_4bet_matchup(opener, bettor3))
            .collect()
    }

    fn extract_4bet_matchup(&self, opener_pos: usize, bettor3_pos: usize) -> SrpMatchup {
        let config = &self.config;
        let mut state = PreflopState::new_6max(config.clone());
        let mut history: Vec<u8> = Vec::new();

        // 1. Fold everyone before opener
        for _ in 0..opener_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // 2. Opener's decision point — find raise action (open)
        let opener_open_history = history.clone();
        let opener_actions = state.actions();
        let open_raise_idx = opener_actions.iter()
            .position(|a| matches!(a, PreflopAction::Raise(_)))
            .expect("opener must have a raise action");

        // 3. Apply opener's raise (open)
        history.push(open_raise_idx as u8);
        state = state.apply(opener_actions[open_raise_idx]);

        // 4. Fold everyone between opener and 3bettor
        for _ in (opener_pos + 1)..bettor3_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // 5. 3bettor's decision point — find raise action (3bet)
        let bettor3_3bet_history = history.clone();
        let bettor3_actions = state.actions();
        let bet3_raise_idx = bettor3_actions.iter()
            .position(|a| matches!(a, PreflopAction::Raise(_)))
            .expect("3bettor must have a raise action");

        // 6. Apply 3bet
        history.push(bet3_raise_idx as u8);
        state = state.apply(bettor3_actions[bet3_raise_idx]);

        // 7. Fold remaining players until opener gets to act again
        while state.to_act as usize != opener_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // 8. Opener facing 3bet — find raise action (4bet)
        let opener_4bet_history = history.clone();
        let opener_facing_actions = state.actions();
        let bet4_raise_idx = opener_facing_actions.iter()
            .position(|a| matches!(a, PreflopAction::Raise(_)))
            .expect("opener must have a raise action for 4bet");

        // 9. Apply 4bet
        history.push(bet4_raise_idx as u8);
        state = state.apply(opener_facing_actions[bet4_raise_idx]);

        // 10. Fold remaining players until 3bettor gets to act again
        while state.to_act as usize != bettor3_pos {
            history.push(0);
            state = state.apply(PreflopAction::Fold);
        }

        // 11. 3bettor facing 4bet — find call action
        let bettor3_call_history = history.clone();
        let bettor3_facing_actions = state.actions();
        let call_idx = bettor3_facing_actions.iter()
            .position(|a| matches!(a, PreflopAction::Call))
            .expect("3bettor must have a call action facing 4bet");

        // 12. Apply call → get pot/stack
        state = state.apply(PreflopAction::Call);
        let pot_chips: i32 = state.bets.iter().sum();
        let eff_stack_chips = state.stacks[opener_pos].min(state.stacks[bettor3_pos]);

        // 13. Extract ranges
        // Opener(4bettor) range = P(open) × P(4bet | facing 3bet)
        let opener_range = self.extract_range_compound(
            &opener_open_history, open_raise_idx,
            &opener_4bet_history, bet4_raise_idx,
        );
        // 3bettor(caller) range = P(3bet | facing open) × P(call | facing 4bet)
        let caller_range = self.extract_range_compound(
            &bettor3_3bet_history, bet3_raise_idx,
            &bettor3_call_history, call_idx,
        );

        SrpMatchup {
            matchup: format!("{} 4bet vs {} call", POSITION_NAMES[opener_pos], POSITION_NAMES[bettor3_pos]),
            pot_chips,
            eff_stack_chips,
            opener: MatchupSide {
                position: POSITION_NAMES[opener_pos].to_string(),
                range: opener_range,
            },
            caller: MatchupSide {
                position: POSITION_NAMES[bettor3_pos].to_string(),
                range: caller_range,
            },
        }
    }

    // -----------------------------------------------------------------------
    // All matchups combined
    // -----------------------------------------------------------------------

    /// Extract all matchup ranges: SRP(15) + 3bet(10) + 4bet(5) = 30 total.
    pub fn extract_all_matchups(&self) -> Vec<SrpMatchup> {
        let mut matchups = self.extract_matchups();
        matchups.extend(self.extract_3bet_matchups());
        matchups.extend(self.extract_4bet_matchups());
        matchups
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card;

    #[test]
    fn test_initial_state() {
        let state = PreflopState::new_6max(PreflopBetConfig::default());
        assert_eq!(state.to_act, 0); // UTG
        assert_eq!(state.bets[4], 1); // SB
        assert_eq!(state.bets[5], 2); // BB
        assert_eq!(state.stacks[4], 199);
        assert_eq!(state.stacks[5], 198);
        assert_eq!(state.stacks[0], 200);
        assert_eq!(state.active_count(), 6);
    }

    #[test]
    fn test_utg_actions() {
        let state = PreflopState::new_6max(PreflopBetConfig::default());
        let actions = state.actions();
        // UTG: Fold, Raise(5) only. No Call (open-limp forbidden), no AllIn (min_allin_depth=1)
        assert_eq!(actions, vec![PreflopAction::Fold, PreflopAction::Raise(5)]);
    }

    #[test]
    fn test_fold_to_bb() {
        let mut state = PreflopState::new_6max(PreflopBetConfig::default());
        // Everyone folds to BB
        for _ in 0..5 {
            state = state.apply(PreflopAction::Fold);
        }
        assert_eq!(state.active_count(), 1);
        assert!(matches!(state.node_type(), PreflopNodeType::TerminalFold(5)));
    }

    #[test]
    fn test_limp_to_bb_option() {
        let mut state = PreflopState::new_6max(PreflopBetConfig::default());
        // UTG-BTN fold
        for _ in 0..4 {
            state = state.apply(PreflopAction::Fold);
        }
        // SB calls (limp)
        state = state.apply(PreflopAction::Call);
        // BB should have option (check or raise)
        assert_eq!(state.to_act, 5);
        assert!(!state.is_closed());
        let actions = state.actions();
        assert!(actions.contains(&PreflopAction::Check));
    }

    #[test]
    fn test_bb_check_terminal() {
        let mut state = PreflopState::new_6max(PreflopBetConfig::default());
        for _ in 0..4 {
            state = state.apply(PreflopAction::Fold);
        }
        state = state.apply(PreflopAction::Call); // SB limps
        state = state.apply(PreflopAction::Check); // BB checks
        assert!(matches!(state.node_type(), PreflopNodeType::TerminalShowdown));
    }

    #[test]
    fn test_open_call_showdown() {
        let mut state = PreflopState::new_6max(PreflopBetConfig::default());
        // UTG opens 2.5bb
        state = state.apply(PreflopAction::Raise(5));
        // HJ-BB fold, except BB calls
        for _ in 0..4 {
            state = state.apply(PreflopAction::Fold);
        }
        // BB calls
        state = state.apply(PreflopAction::Call);
        assert!(matches!(state.node_type(), PreflopNodeType::TerminalShowdown));
    }

    #[test]
    fn test_payoff_fold() {
        let mut state = PreflopState::new_6max(PreflopBetConfig::default());
        state = state.apply(PreflopAction::Raise(5)); // UTG opens 5
        // Everyone folds
        for _ in 0..4 {
            state = state.apply(PreflopAction::Fold);
        }
        state = state.apply(PreflopAction::Fold); // BB folds
        let payoffs = state.payoff_fold(0); // UTG wins
        // UTG invested 5, wins pot of 5 + 1(SB) + 2(BB) = 8, profit = 3
        assert_eq!(payoffs[0], 3.0);
        assert_eq!(payoffs[4], -1.0); // SB lost 1
        assert_eq!(payoffs[5], -2.0); // BB lost 2
    }

    #[test]
    fn test_payoff_showdown_simple() {
        let mut state = PreflopState::new_6max(PreflopBetConfig::default());
        // Set up: everyone folds except SB and BB
        for _ in 0..4 {
            state = state.apply(PreflopAction::Fold);
        }
        state = state.apply(PreflopAction::Call); // SB calls
        state = state.apply(PreflopAction::Check); // BB checks

        // Give SB pocket aces, BB pocket kings
        state.holes[4] = Hand::new().add(card(12, 0)).add(card(12, 1)); // AcAd
        state.holes[5] = Hand::new().add(card(11, 0)).add(card(11, 1)); // KcKd

        // Board: 2h 3h 4s 5s 8d (no straight/flush for anyone)
        let board = Hand::new()
            .add(card(0, 2))  // 2h
            .add(card(1, 2))  // 3h
            .add(card(2, 3))  // 4s
            .add(card(3, 3))  // 5s
            .add(card(6, 1)); // 8d

        let payoffs = state.payoff_showdown(board);
        // SB (AA) wins: pot = 4 (2+2), invested = 2, profit = 2
        assert_eq!(payoffs[4], 2.0);
        assert_eq!(payoffs[5], -2.0);
    }

    #[test]
    fn test_side_pot() {
        let config = PreflopBetConfig {
            min_allin_depth: 0, // Allow open-shove for this test
            ..PreflopBetConfig::default()
        };
        let mut state = PreflopState::new_6max(config);
        // Override stacks: UTG short-stacked at 50 total (including no blind)
        // SB posted 1 (stack=199), BB posted 2 (stack=198), others=200
        state.stacks[0] = 50; // UTG 50 chips behind

        // UTG all-in for 50
        state = state.apply(PreflopAction::AllIn);
        assert_eq!(state.bets[0], 50);

        // HJ-BTN fold
        for _ in 0..3 {
            state = state.apply(PreflopAction::Fold);
        }
        // SB all-in (199 more chips, total bet = 200)
        state = state.apply(PreflopAction::AllIn);
        assert_eq!(state.bets[4], 200); // SB: 1 blind + 199 = 200
        // BB calls (needs 198 more, total bet = 200)
        state = state.apply(PreflopAction::Call);
        assert_eq!(state.bets[5], 200); // BB: 2 blind + 198 = 200

        assert!(matches!(state.node_type(), PreflopNodeType::TerminalShowdown));

        // Give hands: UTG=AA, SB=KK, BB=QQ
        state.holes[0] = Hand::new().add(card(12, 0)).add(card(12, 1));
        state.holes[4] = Hand::new().add(card(11, 0)).add(card(11, 1));
        state.holes[5] = Hand::new().add(card(10, 0)).add(card(10, 1));

        let board = Hand::new()
            .add(card(0, 2)).add(card(1, 2)).add(card(2, 3))
            .add(card(3, 3)).add(card(6, 1));

        let payoffs = state.payoff_showdown(board);
        // Main pot at 50 level: 50*3 = 150, UTG(AA) wins
        // Side pot at 200 level: (200-50)*2 = 300, SB(KK) wins
        // UTG invested 50, gets 150, profit = 100
        // SB invested 200, gets 300, profit = 100
        // BB invested 200, gets 0, loss = -200
        assert_eq!(payoffs[0], 100.0);
        assert_eq!(payoffs[4], 100.0);
        assert_eq!(payoffs[5], -200.0);
    }

    #[test]
    fn test_min_raise_rule() {
        let mut state = PreflopState::new_6max(PreflopBetConfig::default());
        state = state.apply(PreflopAction::Raise(5)); // UTG opens to 5
        // HJ faces: fold, call, 3bet to 18, all-in
        let actions = state.actions();
        assert!(actions.contains(&PreflopAction::Fold));
        assert!(actions.contains(&PreflopAction::Call));
        assert!(actions.contains(&PreflopAction::Raise(18))); // 3-bet
        assert!(actions.contains(&PreflopAction::AllIn));
    }

    #[test]
    fn test_mccfr_basic() {
        let config = PreflopBetConfig::default();
        let mut trainer = PreflopTrainer::new(config, 42);
        trainer.train(1000);
        assert_eq!(trainer.blueprint.iterations, 1000);
        assert!(trainer.blueprint.entries.len() > 0);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let config = PreflopBetConfig::default();
        let mut trainer = PreflopTrainer::new(config, 42);
        trainer.train(100);

        let mut buf = Vec::new();
        trainer.blueprint.save(&mut buf).unwrap();

        let loaded = PreflopBlueprint::load(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.iterations, 100);
        assert_eq!(loaded.entries.len(), trainer.blueprint.entries.len());
    }

    #[test]
    #[ignore] // Long-running: ~30s
    fn test_convergence_1m() {
        use std::time::Instant;

        let config = PreflopBetConfig::default();
        let mut trainer = PreflopTrainer::new(config, 42);

        // Print action labels for UTG RFI
        let state = PreflopState::new_6max(PreflopBetConfig::default());
        let actions = state.actions();
        eprintln!("\nUTG action labels: {:?}", actions);
        eprintln!();

        // Hands to track (index, name)
        let hands = [
            (12u8, "AA"), (11, "KK"), (10, "QQ"),
            (90, "AKs"), (168, "AKo"),
            (0, "22"), (91, "32o"),
        ];

        let batch = 100_000u64;
        let total = 1_000_000u64;
        let start = Instant::now();

        eprintln!("{:>8} | {:>10} | {}", "iter", "iter/sec",
            "AA UTG avg strategy  |  current strategy");
        eprintln!("{}", "-".repeat(120));

        for step in 0..(total / batch) {
            let batch_start = Instant::now();
            trainer.train(batch);
            let batch_elapsed = batch_start.elapsed();
            let ips = batch as f64 / batch_elapsed.as_secs_f64();
            let total_iter = (step + 1) * batch;

            // Print AA convergence curve: both average and current strategy
            let aa_key = PreflopInfoKey { bucket: 12, history: vec![] };
            if let Some(entry) = trainer.blueprint.entries.get(&aa_key) {
                let avg = entry.average_strategy();
                let cur = entry.current_strategy();
                let avg_s: Vec<String> = actions.iter().zip(avg.iter())
                    .map(|(a, &p)| format!("{}={:.1}%", a, p * 100.0))
                    .collect();
                let cur_s: Vec<String> = actions.iter().zip(cur.iter())
                    .map(|(a, &p)| format!("{}={:.1}%", a, p * 100.0))
                    .collect();
                eprintln!("{:>8} | {:>10.0} | {}  |  {}",
                    total_iter, ips, avg_s.join("  "), cur_s.join("  "));
            }
        }

        let total_elapsed = start.elapsed();
        let overall_ips = total as f64 / total_elapsed.as_secs_f64();

        eprintln!();
        eprintln!("=== Performance ===");
        eprintln!("Total: {}s for {} iterations", total_elapsed.as_secs_f64() as u32, total);
        eprintln!("Speed: {:.0} iter/sec", overall_ips);
        eprintln!("Estimated 10M: {:.0}s", 10_000_000.0 / overall_ips);
        eprintln!("Estimated 100M: {:.0}s ({:.1} min)", 100_000_000.0 / overall_ips, 100_000_000.0 / overall_ips / 60.0);
        eprintln!("Info sets: {}", trainer.blueprint.entries.len());

        // Print all tracked hands at final state
        eprintln!();
        eprintln!("=== Final strategies (UTG RFI, 1M iter) ===");
        for &(idx, name) in &hands {
            let key = PreflopInfoKey { bucket: idx, history: vec![] };
            if let Some(entry) = trainer.blueprint.entries.get(&key) {
                let avg = entry.average_strategy();
                let parts: Vec<String> = actions.iter().zip(avg.iter())
                    .map(|(a, &p)| format!("{}={:.1}%", a, p * 100.0))
                    .collect();
                eprintln!("  {:<4}: {}", name, parts.join("  "));
            }
        }

        // Sanity checks
        let aa_key = PreflopInfoKey { bucket: 12, history: vec![] };
        let entry = trainer.blueprint.entries.get(&aa_key).unwrap();
        let avg = entry.average_strategy();
        let fold_prob = avg.get(0).copied().unwrap_or(0.0);
        let raise_prob = avg.get(1).copied().unwrap_or(0.0); // UTG: [Fold, Raise(5)]
        // AA should almost never fold, and should prefer raise
        assert!(fold_prob < 0.05, "AA fold should be <5%, got {:.1}%", fold_prob * 100.0);
        assert!(raise_prob > 0.5, "AA raise should be >50%, got {:.1}%", raise_prob * 100.0);
    }

    /// Heads-up (SB vs BB only) convergence test using dedicated HU trainer.
    /// Small game tree → fast convergence → validates algorithm correctness.
    #[test]
    #[ignore]
    fn test_convergence_heads_up() {
        use std::time::Instant;

        let config = PreflopBetConfig::default();
        let mut trainer = PreflopTrainer::new(config, 42);

        // Get SB action labels from HU initial state
        let state = PreflopState::new_heads_up(PreflopBetConfig::default());
        let sb_actions = state.actions();
        eprintln!("\nHU SB actions: {:?}", sb_actions);

        let batch = 100_000u64;
        let total = 1_000_000u64;
        let start = Instant::now();

        // In HU trainer, SB's history = [] (game starts at SB)
        let sb_history: Vec<u8> = vec![];

        eprintln!();
        eprintln!("{:>8} | {:>8} | {}", "iter", "infosets", "SB AA: avg | current");
        eprintln!("{}", "-".repeat(110));

        for step in 0..(total / batch) {
            let t = Instant::now();
            trainer.train_hu(batch);
            let _ips = batch as f64 / t.elapsed().as_secs_f64();
            let total_iter = (step + 1) * batch;

            let key = PreflopInfoKey { bucket: 12, history: sb_history.clone() };
            if let Some(entry) = trainer.blueprint.entries.get(&key) {
                let avg = entry.average_strategy();
                let cur = entry.current_strategy();
                let avg_s: Vec<String> = sb_actions.iter().zip(avg.iter())
                    .map(|(a, &p)| format!("{}={:.1}%", a, p * 100.0))
                    .collect();
                let cur_s: Vec<String> = sb_actions.iter().zip(cur.iter())
                    .map(|(a, &p)| format!("{}={:.1}%", a, p * 100.0))
                    .collect();
                eprintln!("{:>8} | {:>8} | avg: {}  |  cur: {}",
                    total_iter, trainer.blueprint.entries.len(),
                    avg_s.join("  "), cur_s.join("  "));
            }
        }

        let elapsed = start.elapsed();
        let ips = total as f64 / elapsed.as_secs_f64();
        eprintln!("\nTotal: {:.1}s, {:.0} iter/sec", elapsed.as_secs_f64(), ips);
        eprintln!("Info sets: {}", trainer.blueprint.entries.len());

        // Print final SB RFI for key hands
        eprintln!("\n=== SB vs BB: SB strategy (1M HU iter) ===");
        let hands = [
            (12u8, "AA"), (11, "KK"), (10, "QQ"), (9, "JJ"),
            (90, "AKs"), (168, "AKo"), (89, "QJs"), (167, "QJo"),
            (77, "T9s"), (57, "76s"),
            (0, "22"), (91, "32o"), (155, "T9o"),
        ];
        for &(idx, name) in &hands {
            let key = PreflopInfoKey { bucket: idx, history: sb_history.clone() };
            if let Some(entry) = trainer.blueprint.entries.get(&key) {
                let avg = entry.average_strategy();
                let parts: Vec<String> = sb_actions.iter().zip(avg.iter())
                    .map(|(a, &p)| format!("{}={:.1}%", a, p * 100.0))
                    .collect();
                eprintln!("  {:<4}: {}", name, parts.join("  "));
            } else {
                eprintln!("  {:<4}: (not visited)", name);
            }
        }

        // Print BB response to SB raise for key hands
        // After SB raises (action index 2 = Raise(7)), BB's history = [2]
        let sb_raise_state = state.apply(PreflopAction::Raise(7));
        let bb_actions = sb_raise_state.actions();
        eprintln!("\n=== BB vs SB raise: BB strategy ===");
        eprintln!("BB actions: {:?}", bb_actions);
        let bb_history: Vec<u8> = vec![2]; // SB raised
        for &(idx, name) in &hands {
            let key = PreflopInfoKey { bucket: idx, history: bb_history.clone() };
            if let Some(entry) = trainer.blueprint.entries.get(&key) {
                let avg = entry.average_strategy();
                let parts: Vec<String> = bb_actions.iter().zip(avg.iter())
                    .map(|(a, &p)| format!("{}={:.1}%", a, p * 100.0))
                    .collect();
                eprintln!("  {:<4}: {}", name, parts.join("  "));
            }
        }

        // Assertions
        let key = PreflopInfoKey { bucket: 12, history: sb_history.clone() };
        let entry = trainer.blueprint.entries.get(&key).unwrap();
        let avg = entry.average_strategy();
        let fold = avg[0];
        let raise = avg.get(2).copied().unwrap_or(0.0);
        assert!(fold < 0.02, "SB AA fold < 2%, got {:.1}%", fold * 100.0);
        assert!(raise > 0.3, "SB AA raise > 30%, got {:.1}%", raise * 100.0);
    }
}
