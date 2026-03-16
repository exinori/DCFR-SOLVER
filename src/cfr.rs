/// CFR+ subgame solver — indexed tree with O(N log N) showdown optimization.
///
/// Solves a specific postflop spot given:
/// - Board cards, pot, stacks
/// - OOP and IP ranges (1326-combo weight vectors)
/// - Betting tree (from game.rs)
///
/// Key optimizations:
/// - Pre-built tree with integer node IDs (no HashMap lookups in hot path)
/// - Sort + prefix sum showdown evaluation: O(N log N) instead of O(N²)
/// - Card removal correction via per-card prefix sums
/// - Flat arrays: [1326 * n_actions] for regrets and cumulative strategy
/// - DCFR discounting for 2-3x faster convergence

use crate::card::{combo_from_index, Card, Hand, NUM_COMBOS, CARD_COMBOS};
use crate::eval::{evaluate, Strength};
use crate::game::{Action, BetConfig, GameState, NodeType, Player, Street, TerminalType, OOP, IP};
use crate::iso::{canonical_next_cards, compose_perms};
use crate::range::Range;
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use memmap2::MmapMut;

/// Maximum number of actions at any decision node.
/// Used for fixed-size stack arrays to avoid heap allocation in the hot path.
const MAX_ACTIONS: usize = 8;

thread_local! {
    /// When true, compute_avg_strat_flat uses regret-matched (current) strategy
    /// instead of cumulative average. Used by exploitability_current().
    static FORCE_CURRENT_STRATEGY: Cell<bool> = Cell::new(false);
}

// Regret arena element type: i16 (default, 2 bytes) or i32 (4 bytes, 65536× precision).
// Use `cargo build --features i32-arena` to test with higher precision.
#[cfg(not(feature = "i32-arena"))]
type ArenaInt = i16;
#[cfg(not(feature = "i32-arena"))]
const ARENA_MAX: f32 = 32767.0;

#[cfg(feature = "i32-arena")]
type ArenaInt = i32;
#[cfg(feature = "i32-arena")]
const ARENA_MAX: f32 = 2147483647.0;


// ---------------------------------------------------------------------------
// ComboMap — maps between original 1326-combo indices and compact live indices
// ---------------------------------------------------------------------------

/// Maps between original combo indices (0..1326) and compact "live" indices
/// (0..live_count) where dead combos (blocked by board) are excluded.
///
/// All hot-path arrays (reach, regrets, action_utils, node_util, combo_strats)
/// use compact indexing so inner loops iterate 0..live_count with sequential
/// memory access, preserving SIMD auto-vectorization.
pub struct ComboMap {
    pub live_count: usize,
    /// compact index → original combo index
    pub live_to_combo: Vec<u16>,
    /// original combo index → compact index (-1 if dead)
    pub combo_to_live: [i16; NUM_COMBOS],
}

impl ComboMap {
    /// Build a ComboMap from board cards and player ranges. A combo is excluded
    /// if either card appears on the board OR if neither player has it in range.
    fn new(board: Hand, ranges: &[Range; 2]) -> Self {
        let mut live_to_combo = Vec::with_capacity(NUM_COMBOS);
        let mut combo_to_live = [-1i16; NUM_COMBOS];
        for c in 0..NUM_COMBOS {
            let (c1, c2) = combo_from_index(c as u16);
            if board.contains(c1) || board.contains(c2) {
                continue;
            }
            // Skip combos where neither player has any weight
            if ranges[0].weights[c] == 0.0 && ranges[1].weights[c] == 0.0 {
                continue;
            }
            let li = live_to_combo.len();
            combo_to_live[c] = li as i16;
            live_to_combo.push(c as u16);
        }
        ComboMap {
            live_count: live_to_combo.len(),
            live_to_combo,
            combo_to_live,
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorization-friendly primitives.
// Explicit slice-based ops with asserted bounds help LLVM auto-vectorize.
// ---------------------------------------------------------------------------

/// dst[i] = a[i] * b[i]  (element-wise multiply)
/// #[inline(never)] to force LLVM to vectorize independently of caller complexity.
#[inline(never)]
fn vec_mul(dst: &mut [f32], a: &[f32], b: &[f32]) {
    let n = dst.len();
    assert!(a.len() >= n && b.len() >= n);
    for i in 0..n { dst[i] = a[i] * b[i]; }
}

/// dst[i] += a[i] * b[i]  (fused multiply-accumulate)
#[inline(never)]
fn vec_fma(dst: &mut [f32], a: &[f32], b: &[f32]) {
    let n = dst.len();
    assert!(a.len() >= n && b.len() >= n);
    for i in 0..n { dst[i] += a[i] * b[i]; }
}

/// dst[i] += a[i]  (element-wise add)
#[inline(never)]
fn vec_add(dst: &mut [f32], a: &[f32]) {
    let n = dst.len();
    assert!(a.len() >= n);
    for i in 0..n { dst[i] += a[i]; }
}

/// Initialize the Rayon global thread pool with 16MB stacks.
/// CFR traversal uses large stack arrays ([f32; 1326]) at each recursion level;
/// deep trees (multi-street, max_raises=3) can exceed the default 8MB stack.
pub fn init_rayon_pool() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(16 * 1024 * 1024)
            .build_global()
            .ok();
    });
}

// Thread-local buffer pools to avoid repeated heap allocation in cfr_traverse.
// Buffers are reused across recursive calls and iterations.
thread_local! {
    // SoA layout: strat[action][combo] for cache-friendly, SIMD-friendly access
    static STRAT_POOL: RefCell<Vec<Vec<[f32; NUM_COMBOS]>>> = const { RefCell::new(Vec::new()) };
    static UTILS_POOL: RefCell<Vec<Vec<[f32; NUM_COMBOS]>>> = const { RefCell::new(Vec::new()) };
    static REGRET_POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn pop_strat_buf() -> Vec<[f32; NUM_COMBOS]> {
    STRAT_POOL.with(|pool| {
        pool.borrow_mut().pop()
            .unwrap_or_else(|| vec![[0.0f32; NUM_COMBOS]; MAX_ACTIONS])
    })
}

#[inline]
fn push_strat_buf(buf: Vec<[f32; NUM_COMBOS]>) {
    STRAT_POOL.with(|pool| pool.borrow_mut().push(buf));
}

#[inline]
fn pop_utils_buf() -> Vec<[f32; NUM_COMBOS]> {
    UTILS_POOL.with(|pool| {
        pool.borrow_mut().pop()
            .unwrap_or_else(|| vec![[0.0f32; NUM_COMBOS]; MAX_ACTIONS])
    })
}

#[inline]
fn push_utils_buf(buf: Vec<[f32; NUM_COMBOS]>) {
    UTILS_POOL.with(|pool| pool.borrow_mut().push(buf));
}

#[inline]
fn pop_regret_buf(len: usize) -> Vec<f32> {
    REGRET_POOL.with(|pool| {
        let mut buf = pool.borrow_mut().pop().unwrap_or_else(Vec::new);
        // SAFETY: Callers always overwrite all elements before reading
        // (decode_regrets_i16 writes dst[i] for all i in 0..len).
        // Skipping zeroing is a hot-path optimization.
        #[allow(clippy::uninit_vec)]
        {
            buf.reserve(len);
            unsafe { buf.set_len(len); }
        }
        buf
    })
}

#[inline]
fn push_regret_buf(buf: Vec<f32>) {
    REGRET_POOL.with(|pool| pool.borrow_mut().push(buf));
}

/// Find maximum absolute value in a f32 slice.
/// Uses 4 independent accumulators to break the dependency chain,
/// enabling NEON SIMD auto-vectorization (fabs + fmaxnm on 4 lanes).
#[inline(never)]
fn max_abs_f32(data: &[f32]) -> f32 {
    let n = data.len();
    let mut m0 = 0.0f32;
    let mut m1 = 0.0f32;
    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        m0 = m0.max(data[i].abs());
        m1 = m1.max(data[i + 1].abs());
        m2 = m2.max(data[i + 2].abs());
        m3 = m3.max(data[i + 3].abs());
        i += 4;
    }
    let mut max = m0.max(m1).max(m2).max(m3);
    while i < n {
        max = max.max(data[i].abs());
        i += 1;
    }
    max
}

/// Scale f32 values to ArenaInt with rounding.
#[inline(never)]
fn scale_f32_to_i16(f32_buf: &[f32], out: &mut [ArenaInt], inv: f32) {
    for (o, &r) in out.iter_mut().zip(f32_buf.iter()) {
        *o = (r * inv).round() as ArenaInt;
    }
}

/// Encode f32 regrets to ArenaInt arena. Returns new scale.
#[inline(never)]
fn encode_regrets_i16(f32_buf: &[f32], out: &mut [ArenaInt]) -> f32 {
    debug_assert_eq!(f32_buf.len(), out.len());
    let max_abs = max_abs_f32(f32_buf);
    if max_abs == 0.0 {
        for o in out.iter_mut() { *o = 0; }
        return 0.0;
    }
    let scale = max_abs / ARENA_MAX;
    let inv = 1.0 / scale;
    scale_f32_to_i16(f32_buf, out, inv);
    scale
}

/// Decode ArenaInt arena regrets to f32 buffer.
/// `#[inline(never)]` enables SIMD auto-vectorization for the conversion loop.
#[inline(never)]
fn decode_regrets_i16(i16_slice: &[ArenaInt], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(i16_slice.len(), out.len());
    for (o, &r) in out.iter_mut().zip(i16_slice.iter()) {
        *o = r as f32 * scale;
    }
}

/// CFR+ regret update: regret[i] = max(0, regret[i] + action_util[i] - node_util[i]).
/// Branchless — safe because zero-reach combos have action_util=0 and node_util=0,
/// so the update is a no-op (regret stays ≥ 0 from previous CFR+ clamping).
/// `#[inline(never)]` enables SIMD auto-vectorization (the `t_reach <= 0` branch
/// in the inlined version prevents this).
#[inline(never)]
fn vec_regret_update_cfrplus(regrets: &mut [f32], action_util: &[f32], node_util: &[f32]) {
    for i in 0..regrets.len() {
        regrets[i] = (regrets[i] + action_util[i] - node_util[i]).max(0.0);
    }
}

/// Plain regret update without CFR+ clamping.
#[inline(never)]
fn vec_regret_update(regrets: &mut [f32], action_util: &[f32], node_util: &[f32]) {
    for i in 0..regrets.len() {
        regrets[i] += action_util[i] - node_util[i];
    }
}

/// Clamp all values to non-negative (CFR+ regret flooring).
#[inline(never)]
fn vec_clamp_positive(data: &mut [f32]) {
    for v in data.iter_mut() { *v = v.max(0.0); }
}

/// Branchless regret matching for n_act=2 from pre-decoded f32 buffer.
/// Uses regularized normalization: adds epsilon to avoid division-by-zero,
/// eliminating the `if sum > 0` branch and enabling SIMD auto-vectorization.
/// When all regrets are 0: strat = (eps)/(2*eps) = 0.5 = uniform. Correct.
#[inline(never)]
fn regret_match_2_f32(regrets: &[f32], s0: &mut [f32], s1: &mut [f32], expl_eps: f32, floor: f32) {
    const EPS: f32 = 1e-38;
    let one_m = 1.0 - expl_eps;
    let uni = expl_eps * 0.5;
    let n = s0.len();
    for c in 0..n {
        let r0 = (regrets[c] + floor).max(0.0) + EPS;
        let r1 = (regrets[n + c] + floor).max(0.0) + EPS;
        let inv = 1.0 / (r0 + r1);
        s0[c] = one_m * r0 * inv + uni;
        s1[c] = one_m * r1 * inv + uni;
    }
}

/// Branchless regret matching for n_act=3 from pre-decoded f32 buffer.
#[inline(never)]
fn regret_match_3_f32(regrets: &[f32], s0: &mut [f32], s1: &mut [f32], s2: &mut [f32], expl_eps: f32, floor: f32) {
    const EPS: f32 = 1e-38;
    let one_m = 1.0 - expl_eps;
    let uni = expl_eps / 3.0;
    let n = s0.len();
    for c in 0..n {
        let r0 = (regrets[c] + floor).max(0.0) + EPS;
        let r1 = (regrets[n + c] + floor).max(0.0) + EPS;
        let r2 = (regrets[2 * n + c] + floor).max(0.0) + EPS;
        let inv = 1.0 / (r0 + r1 + r2);
        s0[c] = one_m * r0 * inv + uni;
        s1[c] = one_m * r1 * inv + uni;
        s2[c] = one_m * r2 * inv + uni;
    }
}

/// Batch regret matching for all combos, writing to SoA layout: strat[action][combo].
/// Regrets are i16-encoded in SoA arena layout: [act0_combos..., act1_combos...].
/// Decodes i16→f32 into temp buffer then uses branchless vectorized helpers.
fn regret_match_all(regrets: &[ArenaInt], scale: f32, n_act: usize, live_count: usize, strat: &mut [[f32; NUM_COMBOS]], expl_eps: f32, floor: f32) {
    // Decode i16→f32 first, then dispatch to branchless vectorized helpers.
    let regret_len = n_act * live_count;
    let mut buf = pop_regret_buf(regret_len);
    decode_regrets_i16(regrets, scale, &mut buf);

    if n_act == 2 {
        let (s0, rest) = strat.split_at_mut(1);
        regret_match_2_f32(&buf[..regret_len], &mut s0[0][..live_count], &mut rest[0][..live_count], expl_eps, floor);
    } else if n_act == 3 {
        let (s0, rest) = strat.split_at_mut(1);
        let (s1, rest) = rest.split_at_mut(1);
        regret_match_3_f32(&buf[..regret_len], &mut s0[0][..live_count], &mut s1[0][..live_count], &mut rest[0][..live_count], expl_eps, floor);
    } else {
        const EPS: f32 = 1e-38;
        let one_m = 1.0 - expl_eps;
        let uni = expl_eps / n_act as f32;
        for c in 0..live_count {
            let mut pos_sum = 0.0f32;
            for a in 0..n_act {
                let r = (buf[a * live_count + c] + floor).max(0.0) + EPS;
                strat[a][c] = r;
                pos_sum += r;
            }
            let inv = 1.0 / pos_sum;
            for a in 0..n_act { strat[a][c] = one_m * strat[a][c] * inv + uni; }
        }
    }
    push_regret_buf(buf);
}

/// Softmax regret matching for FTRL-style MaxEnt: σ(a) = exp(R(a)/τ) / Σ exp(R(a)/τ).
/// Unlike RM+ which clips negative regrets, softmax smoothly distributes probability
/// based on regret magnitude, naturally selecting MaxEnt NE for indifferent combos.
#[inline(never)]
fn softmax_match_all(regrets: &[ArenaInt], scale: f32, n_act: usize, live_count: usize, strat: &mut [[f32; NUM_COMBOS]], temp: f32) {
    let regret_len = n_act * live_count;
    let mut buf = pop_regret_buf(regret_len);
    decode_regrets_i16(regrets, scale, &mut buf);

    let inv_temp = 1.0 / temp;
    for c in 0..live_count {
        // Find max regret for numerical stability (log-sum-exp trick)
        let mut max_r = f32::NEG_INFINITY;
        for a in 0..n_act {
            max_r = max_r.max(buf[a * live_count + c]);
        }
        // exp((r - max_r) / τ) for each action
        let mut sum_exp = 0.0f32;
        for a in 0..n_act {
            let e = ((buf[a * live_count + c] - max_r) * inv_temp).exp();
            strat[a][c] = e;
            sum_exp += e;
        }
        let inv = 1.0 / sum_exp;
        for a in 0..n_act {
            strat[a][c] *= inv;
        }
    }
    push_regret_buf(buf);
}

/// Softmax from pre-decoded f32 regrets (for fused decode path).
#[inline(never)]
fn softmax_match_f32(regrets: &[f32], n_act: usize, live_count: usize, strat: &mut [[f32; NUM_COMBOS]], temp: f32) {
    let inv_temp = 1.0 / temp;
    for c in 0..live_count {
        let mut max_r = f32::NEG_INFINITY;
        for a in 0..n_act {
            max_r = max_r.max(regrets[a * live_count + c]);
        }
        let mut sum_exp = 0.0f32;
        for a in 0..n_act {
            let e = ((regrets[a * live_count + c] - max_r) * inv_temp).exp();
            strat[a][c] = e;
            sum_exp += e;
        }
        let inv = 1.0 / sum_exp;
        for a in 0..n_act {
            strat[a][c] *= inv;
        }
    }
}

/// Regret matching from pre-decoded f32 regrets (SoA layout) when decode is fused.
/// Dispatches to branchless vectorized helpers.
fn regret_match_f32(regrets: &[f32], n_act: usize, live_count: usize, strat: &mut [[f32; NUM_COMBOS]], expl_eps: f32, floor: f32) {
    if n_act == 2 {
        let (s0, rest) = strat.split_at_mut(1);
        regret_match_2_f32(regrets, &mut s0[0][..live_count], &mut rest[0][..live_count], expl_eps, floor);
    } else if n_act == 3 {
        let (s0, rest) = strat.split_at_mut(1);
        let (s1, rest) = rest.split_at_mut(1);
        regret_match_3_f32(regrets, &mut s0[0][..live_count], &mut s1[0][..live_count], &mut rest[0][..live_count], expl_eps, floor);
    } else {
        const EPS: f32 = 1e-38;
        let one_m = 1.0 - expl_eps;
        let uni = expl_eps / n_act as f32;
        for c in 0..live_count {
            let mut pos_sum = 0.0f32;
            for a in 0..n_act {
                let r = (regrets[a * live_count + c] + floor).max(0.0) + EPS;
                strat[a][c] = r;
                pos_sum += r;
            }
            let inv = 1.0 / pos_sum;
            for a in 0..n_act { strat[a][c] = one_m * strat[a][c] * inv + uni; }
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// DCFR discounting mode.
#[derive(Clone, Debug, PartialEq)]
pub enum DcfrMode {
    /// Standard DCFR: alpha=1.5, beta=0, gamma=2.0.
    Standard,
    /// HS-DCFR(30) from "Faster Game Solving via Hyperparameter Schedules" (2024).
    /// Dynamic alpha/beta/gamma that vary with iteration t.
    HsSchedule,
    /// b-inary trick: gamma=3.0 + cum_strategy reset at powers of 4.
    BinaryTrick,
    /// LCFR (Linear CFR): alpha=1.0, beta=0.0, gamma=1.0.
    /// Linear weighting of regrets and strategies. May converge to different NE.
    Linear,
    /// Custom DCFR parameters: (alpha, beta, gamma).
    Custom(f64, f64, f64),
}

#[derive(Clone)]
pub struct SubgameConfig {
    pub board: Hand,
    pub pot: i32,
    pub stacks: [i32; 2],
    pub ranges: [Range; 2],
    pub iterations: u32,
    pub street: Street,
    /// Fraction of iterations to skip cum_strategy accumulation (0.0–1.0).
    pub warmup_frac: f32,
    /// Custom bet sizes. None = use defaults.
    pub bet_config: Option<Arc<BetConfig>>,
    /// Enable DCFR discounting (default true).
    pub dcfr: bool,
    /// Use CFR+ regret flooring at 0 (default true).
    /// When false, uses vanilla CFR (negative regrets accumulate).
    /// Vanilla CFR may converge better for multi-street games.
    pub cfr_plus: bool,
    /// Skip cum_strategy allocation entirely. When true, average strategy
    /// is computed from the final regret-matched strategy instead.
    /// Saves ~50% memory. Safe with DCFR at 300+ iterations.
    pub skip_cum_strategy: bool,
    /// DCFR mode. Default: Standard.
    pub dcfr_mode: DcfrMode,
    /// Depth limit: stop at this street and use NN evaluation.
    /// None = full solve (default). Some(Street::River) = solve up to river, NN for river.
    pub depth_limit: Option<Street>,
    /// Rake percentage (0.05 = 5%). Default 0.0 (no rake).
    pub rake_pct: f32,
    /// Rake cap in chips. Default 0 (no cap).
    pub rake_cap: f32,
    /// Exploration epsilon for regret matching.
    /// strategy = (1-eps) * RM+(regret) + eps / n_actions.
    /// Encourages mixing on indifferent hands. 0.0 = disabled (default).
    pub exploration_eps: f32,
    /// Legacy entropy bonus (τ) for regret-based MaxEnt regularization.
    /// Adds τ * (-log(σ(a))) to regrets. 0.0 = disabled.
    pub entropy_bonus: f32,
    /// Linearly anneal entropy_bonus to 0 over iterations.
    /// Start with high τ (exploration), decay to 0 (convergence).
    pub entropy_anneal: bool,
    /// Apply entropy bonus only at the root decision node (OOP's first action).
    /// Limits exploitability impact to a single node while encouraging donk mixing.
    pub entropy_root_only: bool,
    /// Softmax temperature for FTRL-style regret matching.
    /// Replaces RM+ (max(R,0)/Σ) with softmax (exp(R/τ)/Σexp(R/τ)).
    /// Naturally selects MaxEnt NE among indifferent combos.
    /// 0.0 = disabled (use standard RM+). Try 1.0, 5.0, 10.0.
    pub softmax_temp: f32,
    /// Current iteration (set by solver each iteration, used for QRE temp scaling).
    pub current_iteration: u32,
    /// Disable suit isomorphism (enumerate all cards individually at chance nodes).
    /// For debugging — much slower but removes ISO as a variable.
    pub use_iso: bool,
    /// Diluted best response: mix opponent strategy with uniform.
    /// σ_opp = (1-δ) * σ_opp + δ * uniform.
    /// Makes the traverser hedge against opponent mistakes.
    /// 0.0 = disabled. Try 0.01, 0.05, 0.1.
    pub opp_dilute: f32,
    /// Regret matching floor: adds a decaying positive offset to regrets before clamping.
    /// Prevents early CFR+ clamping from permanently killing actions (e.g., donk bets).
    /// Effective floor = rm_floor / (1 + 0.01 * iteration). 0.0 = disabled (default).
    pub rm_floor: f32,
    /// Alternating updates: only traverse one player per iteration (OOP on odd, IP on even).
    /// Matches Tammelin's original CFR+ paper. Default false (simultaneous).
    pub alternating: bool,
    /// Linear weighting on cum_strategy: multiply contribution by iteration t.
    /// Matches standard CFR+ / LCFR averaging. Default false (uniform weighting).
    pub t_weight: bool,
    /// Frozen root strategy: if set, root node (cfr_idx=0) uses these fixed
    /// Frozen root: pin root OOP strategy to external data.
    /// Layout: [original_combo_index] = bet fraction (0.0 to 1.0).
    /// During CFR, root uses these fractions instead of regret matching, skips regret updates.
    /// Used for 2-stage equilibrium selection: freeze root mixing, let subtrees re-adapt.
    pub frozen_root: Option<Box<[f32; NUM_COMBOS]>>,
    /// Seed root check regret with a positive bias to push toward passive NE.
    /// 0.0 = disabled. Try 0.1, 1.0, 10.0.
    pub check_bias: f32,
    /// Preference-CFR delta for passive actions (action 0 = check/fold).
    /// Multiplies action 0's positive regret in RM step: σ(a0) = δ·R⁺(a0) / (δ·R⁺(a0) + Σ R⁺(ai)).
    /// δ > 1.0 biases toward passive NE. 1.0 = disabled (default). Try 2.0, 3.0, 5.0.
    /// Unlike check_bias, this preserves regret accumulation and NE convergence (Preference-CFR, 2024).
    pub pref_passive_delta: f32,
    /// Root check reward: pot-relative epsilon for passive NE selection.
    /// Adds `pref_beta * pot` to check action's counterfactual value at root only.
    /// Indifferent hands prefer check → passive NE (closer to GTO+/Pio).
    /// 0.0 = disabled (default). Try 7.0-8.0 for GTO+ matching.
    pub pref_beta: f32,
    /// Apply check reward to all decision nodes (not just root).
    pub pref_beta_all_nodes: bool,
    /// Regret-Based Pruning (RBP): skip subtrees of zero-strategy actions at traverser nodes.
    /// After warm-up (100 iter), actions where all combos have 0 strategy are not traversed.
    /// Every 100 iterations, a full (unpruned) traversal ensures pruned actions can recover.
    /// Only works with CFR+ (regrets clamped ≥ 0). Default false.
    pub pruning: bool,
    /// Per-combo check bias at root: indexed by original combo index (0..1326).
    /// Adds bias[orig] to check action's counterfactual value for each combo.
    /// Positive = encourage check, negative = encourage bet.
    /// Used for hand-specific NE selection (calibrated from GTO+ target).
    pub combo_check_bias: Option<Box<[f32; NUM_COMBOS]>>,
    /// Frozen root warm-up: run first N iterations with frozen_root, then unfreeze.
    /// 0 = disabled (default). Only effective when frozen_root is set.
    pub frozen_warmup: u32,
    /// Decay factor for root cum_strategy at unfreeze time. 1.0 = keep all (default),
    /// 0.0 = full reset, 0.1 = keep 10%. Lower values let post-unfreeze iterations
    /// dominate the average strategy, improving exploitability at the cost of diff.
    pub unfreeze_decay: f32,
}

// ---------------------------------------------------------------------------
// Tree nodes (immutable during CFR iterations)
// ---------------------------------------------------------------------------

pub enum TreeNode {
    Decision {
        player: Player,
        actions: Vec<Action>,
        children: Vec<u32>,   // tree node IDs, one per action
        cfr_idx: u32,         // index into CfrData vec
    },
    Chance {
        /// Each canonical card gets its own subtree with separate CFR data.
        /// Isomorphic cards share the same subtree via combo permutations.
        children: Vec<(u8, u32)>,          // (canonical_card, child tree node ID)
        /// For each child, the combo permutations for its isomorphic cards.
        /// iso_perms[i] is empty when the card has no isomorphic partners.
        iso_perms: Vec<Vec<[u16; NUM_COMBOS]>>,
        /// Actual number of available cards (for normalization).
        n_actual: u8,
    },
    TerminalFold(Player),
    TerminalShowdown,
    /// Depth-limited leaf: NN evaluates remaining streets.
    /// vn_ptr is a raw pointer to ValueNet (kept alive by SubgameSolver's Arc).
    DepthLimitedLeaf { vn_ptr: usize },
}

// ---------------------------------------------------------------------------
// RegretArena — mmap-backed contiguous ArenaInt storage for all regret data
// ---------------------------------------------------------------------------

pub struct RegretArena {
    ptr: *mut ArenaInt,
    len: usize, // total element count
    _mmap: Option<MmapMut>,
}

impl RegretArena {
    /// Create an empty arena (zero size).
    fn empty() -> Self {
        RegretArena {
            ptr: std::ptr::null_mut(),
            len: 0,
            _mmap: None,
        }
    }

    /// Create an arena with `total_elements` ArenaInt elements, zero-initialized.
    pub(crate) fn new(total_elements: usize) -> Self {
        if total_elements == 0 {
            return Self::empty();
        }
        let byte_len = total_elements * std::mem::size_of::<ArenaInt>();

        // Anonymous mmap: OS-backed, no file I/O overhead.
        // OS handles swap/paging under memory pressure (same OOM protection as file-backed).
        let mut mmap = MmapMut::map_anon(byte_len)
            .expect("failed to create anonymous mmap for RegretArena");

        // map_anon is zero-initialized by the OS.

        // Linux: advise hugepages for TLB efficiency
        #[cfg(target_os = "linux")]
        {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    byte_len,
                    libc::MADV_HUGEPAGE,
                );
            }
        }

        let ptr = mmap.as_mut_ptr() as *mut ArenaInt;
        RegretArena {
            ptr,
            len: total_elements,
            _mmap: Some(mmap),
        }
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut ArenaInt { self.ptr }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[ArenaInt] {
        if self.len == 0 { return &[]; }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [ArenaInt] {
        if self.len == 0 { return &mut []; }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

unsafe impl Send for RegretArena {}
unsafe impl Sync for RegretArena {}

// ---------------------------------------------------------------------------
// CFR data (mutable during iterations)
// ---------------------------------------------------------------------------

pub struct CfrData {
    pub n_actions: usize,
    pub live_count: usize,              // number of live combos (from ComboMap)
    pub regret_offset: usize,           // offset into RegretArena (in ArenaInt units)
    pub regret_scale: Cell<f32>,        // ArenaInt→f32 decode: f32_val = val * scale
    pub cum_strategy: Option<Vec<f32>>, // [live_count * n_actions], lazy-allocated
}

// SAFETY: CfrData is only accessed with disjoint indices across threads.
// The Cell<f32> field uses interior mutability for regret_scale, which is
// only written by the owning traverser thread for each node.
unsafe impl Sync for CfrData {}

impl CfrData {
    fn new(n_actions: usize, live_count: usize, regret_offset: usize) -> Self {
        CfrData {
            n_actions,
            live_count,
            regret_offset,
            regret_scale: Cell::new(0.0),
            cum_strategy: None,
        }
    }

    pub(crate) fn ensure_cum_strategy(&mut self) {
        if self.cum_strategy.is_none() {
            self.cum_strategy = Some(vec![0.0; self.live_count * self.n_actions]);
        }
    }

    /// Number of i16 elements this node uses in the arena.
    #[inline]
    fn regret_len(&self) -> usize {
        self.live_count * self.n_actions
    }

    fn average_strategy(&self, combo: usize, arena: &[ArenaInt], expl_eps: f32, softmax_temp: f32) -> Vec<f32> {
        let n = self.n_actions;
        let mut strat = vec![0.0f32; n];
        if let Some(ref cum) = self.cum_strategy {
            let mut total = 0.0f32;
            for a in 0..n {
                strat[a] = cum[combo * n + a];
                total += strat[a];
            }
            if total > 0.0 {
                for s in &mut strat { *s /= total; }
            } else {
                strat.fill(1.0 / n as f32);
            }
        } else if softmax_temp > 0.0 {
            // QRE mode: softmax of cumulative regrets
            let regrets = &arena[self.regret_offset..self.regret_offset + self.regret_len()];
            let scale = self.regret_scale.get();
            let lc = self.live_count;
            let inv_temp = 1.0 / softmax_temp;
            let mut max_r = f32::NEG_INFINITY;
            for a in 0..n {
                let r = regrets[a * lc + combo] as f32 * scale;
                if r > max_r { max_r = r; }
            }
            let mut sum_exp = 0.0f32;
            for a in 0..n {
                let r = regrets[a * lc + combo] as f32 * scale;
                let e = ((r - max_r) * inv_temp).exp();
                strat[a] = e;
                sum_exp += e;
            }
            let inv = 1.0 / sum_exp;
            for s in &mut strat { *s *= inv; }
        } else {
            // Fallback: regret-matched strategy from current i16 regrets (SoA layout).
            let regrets = &arena[self.regret_offset..self.regret_offset + self.regret_len()];
            let scale = self.regret_scale.get();
            let lc = self.live_count;
            let mut pos_sum = 0.0f32;
            for a in 0..n {
                let r = (regrets[a * lc + combo] as f32 * scale).max(0.0);
                strat[a] = r;
                pos_sum += r;
            }
            if pos_sum > 0.0 {
                let inv = 1.0 / pos_sum;
                for s in &mut strat { *s *= inv; }
            } else {
                strat.fill(1.0 / n as f32);
            }
        }
        // Apply exploration epsilon to output
        if expl_eps > 0.0 {
            let one_m = 1.0 - expl_eps;
            let uni = expl_eps / n as f32;
            for s in &mut strat { *s = one_m * *s + uni; }
        }
        strat
    }
}

// ---------------------------------------------------------------------------
// SubgameSolver
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Precomputed showdown data (board-specific, immutable during CFR)
// ---------------------------------------------------------------------------

/// Precomputed combo strengths for a given board, sorted by strength.
pub(crate) struct BoardShowdown {
    /// All live combos sorted by strength (ascending).
    /// (combo_index, strength, card1, card2)
    sorted: Vec<(u16, Strength, Card, Card)>,
}

impl BoardShowdown {
    /// Compute showdown data for a given board.
    /// `combo_map` provides the flop-level compact mapping.
    /// For turn/river boards with additional dead cards, combos blocked by
    /// those extra cards are still present in the sorted list but will have
    /// reach=0 (and be skipped during utility computation).
    fn compute(board: Hand, combo_map: &ComboMap) -> Self {
        let mut sorted = Vec::new();

        for i in 0..combo_map.live_count {
            let orig = combo_map.live_to_combo[i] as usize;
            let (c1, c2) = combo_from_index(orig as u16);
            // Turn/river cards beyond flop may block additional combos
            if board.contains(c1) || board.contains(c2) { continue; }
            let hand = board.union(Hand::from_card(c1).add(c2));
            let strength = evaluate(hand);
            sorted.push((i as u16, strength, c1, c2));
        }
        sorted.sort_by(|a, b| a.1.cmp(&b.1));

        BoardShowdown { sorted }
    }
}

/// Cache of precomputed showdown data, keyed by board (u64 bits).
pub(crate) struct ShowdownCache {
    entries: FxHashMap<u64, BoardShowdown>,
}

impl ShowdownCache {
    fn new() -> Self {
        ShowdownCache { entries: FxHashMap::default() }
    }

    fn get_or_compute(&mut self, board: Hand, combo_map: &ComboMap) -> &BoardShowdown {
        let key = board.0;
        self.entries.entry(key).or_insert_with(|| BoardShowdown::compute(board, combo_map))
    }

    fn get(&self, board: Hand) -> Option<&BoardShowdown> {
        self.entries.get(&board.0)
    }
}

pub struct SubgameSolver {
    pub config: SubgameConfig,
    pub(crate) tree: Vec<TreeNode>,
    pub(crate) cfr: Vec<CfrData>,
    pub(crate) arena: RegretArena,
    /// Maps action sequence → tree node ID (for public API lookups only).
    node_index: FxHashMap<Vec<Action>, u32>,
    /// Precomputed showdown data per board.
    pub(crate) sd_cache: ShowdownCache,
    pub(crate) root_id: u32,
    pub(crate) iteration: u32,
    /// Maps between original 1326 combo indices and compact live indices.
    pub(crate) combo_map: ComboMap,
    /// Neural network for depth-limited solving (keeps Arc alive for raw ptrs in tree).
    #[cfg(feature = "nn")]
    valuenet: Option<Arc<crate::valuenet::ValueNet>>,
}

impl SubgameSolver {
    pub fn new(config: SubgameConfig) -> Self {
        init_rayon_pool();
        let combo_map = ComboMap::new(config.board, &config.ranges);
        SubgameSolver {
            config,
            tree: Vec::new(),
            cfr: Vec::new(),
            arena: RegretArena::empty(),
            node_index: FxHashMap::default(),
            sd_cache: ShowdownCache::new(),
            root_id: 0,
            iteration: 0,
            combo_map,
            #[cfg(feature = "nn")]
            valuenet: None,
        }
    }

    /// Set the value network for depth-limited solving.
    #[cfg(feature = "nn")]
    pub fn set_valuenet(&mut self, vn: Arc<crate::valuenet::ValueNet>) {
        self.valuenet = Some(vn);
    }

    /// Number of decision nodes in the tree.
    pub fn num_decision_nodes(&self) -> usize {
        self.cfr.len()
    }

    /// Access the combo map (for building frozen_root externally).
    pub fn combo_map(&self) -> &ComboMap {
        &self.combo_map
    }

    /// Number of actions at the root node.
    pub fn root_n_actions(&self) -> usize {
        match &self.tree[self.root_id as usize] {
            TreeNode::Decision { actions, .. } => actions.len(),
            _ => 0,
        }
    }

    /// Count combos with non-zero range weight per player (diagnostic).
    pub fn per_player_active_counts(&self) -> (usize, usize, usize, usize) {
        let cm = &self.combo_map;
        let mut oop_active = 0usize;
        let mut ip_active = 0usize;
        let mut both_active = 0usize;
        let mut either_active = 0usize;
        for i in 0..cm.live_count {
            let orig = cm.live_to_combo[i] as usize;
            let oop = self.config.ranges[0].weights[orig] > 0.0;
            let ip = self.config.ranges[1].weights[orig] > 0.0;
            if oop { oop_active += 1; }
            if ip { ip_active += 1; }
            if oop && ip { both_active += 1; }
            if oop || ip { either_active += 1; }
        }
        (oop_active, ip_active, both_active, either_active)
    }

    /// Total number of tree nodes (decision + chance + terminal).
    pub fn tree_node_count(&self) -> usize {
        self.tree.len()
    }

    /// Create root GameState from config, including bet_config if set.
    pub(crate) fn root_state(&self) -> GameState {
        let mut state = GameState::new_postflop(
            self.config.street,
            self.config.pot,
            self.config.stacks,
            self.config.board,
        );
        if let Some(ref bc) = self.config.bet_config {
            state.bet_config = Some(bc.clone());
        }
        state
    }

    /// Run CFR+ for configured iterations.
    pub fn solve(&mut self) {
        self.solve_with_callback(|_, _| {});
    }

    /// Run CFR+ with progress callback at doubling intervals.
    pub fn solve_with_callback<F: FnMut(u32, &SubgameSolver)>(&mut self, callback: F) {
        self.solve_with_report_interval(0, callback);
    }

    /// Run CFR+ with progress callback at a fixed interval.
    /// `every` = 0 means doubling intervals (1,2,4,8,...); `every` > 0 means every N iterations.
    pub fn solve_with_report_interval<F: FnMut(u32, &SubgameSolver)>(&mut self, every: u32, mut callback: F) {
        let mut reach = self.initial_reach();
        let root_state = self.root_state();

        // Phase 1a: Build tree (CfrData created with placeholder offset=0)
        {
            let mut action_seq = Vec::with_capacity(20);
            self.root_id = self.build_tree(&root_state, &mut action_seq);
        }

        // Phase 1b: Allocate arena and assign offsets (compact: live_count * n_actions per node)
        {
            let total_elements: usize = self.cfr.iter()
                .map(|d| d.live_count * d.n_actions)
                .sum();
            self.arena = RegretArena::new(total_elements);
            let mut offset = 0usize;
            for data in self.cfr.iter_mut() {
                data.regret_offset = offset;
                offset += data.live_count * data.n_actions;
            }
        }

        // Phase 1c: Remap iso_perms from original→original to compact→compact indices
        self.remap_iso_perms();

        // Phase 1d: Pre-populate showdown cache for all reachable boards.
        self.precompute_showdowns();

        // Phase 1e-pre: Seed root regrets from target strategy (e.g., GTO+ frequencies).
        // Sets R(action_i) = target_fraction_i * seed_strength for each combo.
        // This initializes CFR to start from a specific strategy, then let normal
        // regret updates confirm/adjust it. If the target is a valid NE, it stays stable.
        if let Some(ref target) = self.config.frozen_root {
            if self.config.frozen_warmup == u32::MAX && !self.cfr.is_empty() {
                // Special mode: frozen_warmup=MAX means "seed regrets, don't freeze"
                let root = &self.cfr[0];
                let n_actions = root.n_actions;
                let live_count = root.live_count;
                let offset = root.regret_offset;
                let regret_len = n_actions * live_count;
                let mut buf = vec![0.0f32; regret_len];
                let seed_strength = 500.0f32; // Large enough to resist early drift
                let cm = &self.combo_map;
                for c in 0..live_count {
                    let orig = cm.live_to_combo[c] as usize;
                    let bet_frac = target[orig]; // 0.0 to 1.0
                    let chk_frac = 1.0 - bet_frac;
                    // action 0 = Check, action 1 = Bet
                    buf[0 * live_count + c] = chk_frac * seed_strength;
                    if n_actions > 1 {
                        buf[1 * live_count + c] = bet_frac * seed_strength;
                    }
                }
                let arena_slice = self.arena.as_mut_slice();
                let scale = encode_regrets_i16(&buf, &mut arena_slice[offset..offset + regret_len]);
                self.cfr[0].regret_scale.set(scale);
                println!("  Seeded root regrets from target strategy (strength={:.0})", seed_strength);
                // Clear frozen_root so normal solving proceeds
                self.config.frozen_root = None;
            }
        }

        // Phase 1e: Seed root check regret (bias toward passive NE).
        if self.config.check_bias > 0.0 && !self.cfr.is_empty() {
            let root = &self.cfr[0];
            let n_actions = root.n_actions;
            let live_count = root.live_count;
            let offset = root.regret_offset;
            let bias = self.config.check_bias;
            // Build f32 regret buffer: check (action 0) gets bias, others get 0
            let regret_len = n_actions * live_count;
            let mut buf = vec![0.0f32; regret_len];
            for c in 0..live_count {
                buf[0 * live_count + c] = bias; // action 0 = Check
            }
            let arena_slice = self.arena.as_mut_slice();
            let scale = encode_regrets_i16(&buf, &mut arena_slice[offset..offset + regret_len]);
            self.cfr[0].regret_scale.set(scale);
            println!("  Seeded root check regret with bias={:.1}", bias);
        }

        let warmup_iters = (self.config.warmup_frac * self.config.iterations as f32) as u32;
        let total = self.config.iterations;
        let mut next_report = if every > 0 { every } else { 1u32 };

        // Phase 2: CFR+ iterations
        let arena_ptr = self.arena.as_ptr();
        for _ in 0..total {
            self.iteration += 1;
            self.config.current_iteration = self.iteration;

            // Frozen root warm-up: unfreeze after N iterations.
            // Keep cum_strategy: frozen phase biases average toward GTO+ direction.
            if self.config.frozen_warmup > 0
                && self.iteration == self.config.frozen_warmup + 1
                && self.config.frozen_root.is_some()
            {
                let warmup_n = self.config.frozen_warmup;
                println!("  Unfreezing root at iteration {} (keeping cum_strategy)", warmup_n);

                // Seed root regrets from GTO+ fractions so solver continues in GTO+ direction.
                // Without this, root regrets are all zero after frozen phase → uniform strategy.
                {
                    let target = self.config.frozen_root.as_ref().unwrap();
                    let root = &self.cfr[0];
                    let n_actions = root.n_actions;
                    let live_count = root.live_count;
                    let offset = root.regret_offset;
                    let regret_len = n_actions * live_count;
                    let mut buf = vec![0.0f32; regret_len];
                    // Seed strength proportional to warmup iterations so it resists drift.
                    let seed_strength = warmup_n as f32;
                    let cm = &self.combo_map;
                    for c in 0..live_count {
                        let orig = cm.live_to_combo[c] as usize;
                        let bet_frac = target[orig].clamp(0.0, 1.0);
                        let chk_frac = 1.0 - bet_frac;
                        buf[0 * live_count + c] = chk_frac * seed_strength;
                        if n_actions > 1 {
                            buf[1 * live_count + c] = bet_frac * seed_strength;
                        }
                    }
                    let arena_slice = self.arena.as_mut_slice();
                    let scale = encode_regrets_i16(&buf, &mut arena_slice[offset..offset + regret_len]);
                    self.cfr[0].regret_scale.set(scale);
                    println!("  Seeded root regrets at unfreeze (strength={:.0})", seed_strength);
                }

                // Scale down root cum_strategy so post-unfreeze iterations have more weight.
                // Factor = unfreeze_decay (CLI param). 0.0 = full reset, 1.0 = keep all.
                let decay = self.config.unfreeze_decay;
                if decay < 1.0 {
                    if let Some(ref mut cum) = self.cfr[0].cum_strategy {
                        for v in cum.iter_mut() {
                            *v *= decay;
                        }
                    }
                    println!("  Decayed root cum_strategy by {:.2}", decay);
                }

                self.config.frozen_root = None;
            }

            if !self.config.skip_cum_strategy && self.iteration == warmup_iters + 1 {
                for data in self.cfr.iter_mut() {
                    data.ensure_cum_strategy();
                }
            }

            let traversers: Vec<Player> = if self.config.alternating {
                vec![if self.iteration % 2 == 1 { OOP } else { IP }]
            } else {
                vec![OOP, IP]
            };
            for traverser in traversers {
                let mut _discard = [0.0f32; NUM_COMBOS];
                if self.config.skip_cum_strategy {
                    let dummy = [0.0f32; NUM_COMBOS];
                    cfr_traverse(
                        &self.tree, &self.cfr, arena_ptr, &self.config, &self.sd_cache,
                        &self.combo_map,
                        &root_state, &mut reach, &dummy, traverser, self.root_id,
                        &[], &mut _discard,
                    );
                } else {
                    let pure_t = reach[traverser as usize];
                    cfr_traverse(
                        &self.tree, &self.cfr, arena_ptr, &self.config, &self.sd_cache,
                        &self.combo_map,
                        &root_state, &mut reach, &pure_t, traverser, self.root_id,
                        &[], &mut _discard,
                    );
                }
            }

            if self.config.dcfr {
                discount_all(&mut self.cfr, self.iteration,
                    &self.config.dcfr_mode, self.config.cfr_plus, arena_ptr);
            }

            // b-inary trick: reset cum_strategy at powers of 4
            if self.config.dcfr_mode == DcfrMode::BinaryTrick && !self.config.skip_cum_strategy {
                let t = self.iteration;
                if t > 1 && t.is_power_of_two() {
                    let log2 = t.trailing_zeros();
                    if log2 % 2 == 0 { // power of 4: 4, 16, 64, 256, 1024, ...
                        for data in self.cfr.iter_mut() {
                            if let Some(ref mut cum) = data.cum_strategy {
                                for s in cum.iter_mut() { *s = 0.0; }
                            }
                        }
                    }
                }
            }

            if self.iteration >= next_report || self.iteration == total {
                callback(self.iteration, self);
                if every > 0 {
                    next_report += every;
                } else {
                    while next_report <= self.iteration {
                        next_report = next_report.saturating_mul(2);
                    }
                }
            }
        }
    }

    /// Remap iso_perms from original combo indices to compact indices.
    /// After this, iso_perms[i][perm][compact_c] = compact index of permuted combo.
    ///
    /// Some live combos may map to flop-dead combos under suit permutation (because
    /// the perm may swap suits that appear on the board). In those cases, the permuted
    /// combo would have had 0 utility in the canonical subtree anyway (0 reach), so we
    /// map to `live_count` which always contains 0 in the [NUM_COMBOS]-sized util arrays.
    pub(crate) fn remap_iso_perms(&mut self) {
        let cm = &self.combo_map;
        for node in &mut self.tree {
            if let TreeNode::Chance { iso_perms, .. } = node {
                for card_perms in iso_perms.iter_mut() {
                    for perm in card_perms.iter_mut() {
                        let mut compact_perm = [0u16; NUM_COMBOS];
                        for i in 0..cm.live_count {
                            let orig_c = cm.live_to_combo[i] as usize;
                            let permuted_orig = perm[orig_c] as usize;
                            let compact_target = cm.combo_to_live[permuted_orig];
                            if compact_target >= 0 {
                                compact_perm[i] = compact_target as u16;
                            } else {
                                // Perm maps to a flop-dead combo: map to live_count (zero-padded slot)
                                compact_perm[i] = cm.live_count as u16;
                            }
                        }
                        // Identity for dead slots (compose_perms iterates 0..NUM_COMBOS)
                        for i in cm.live_count..NUM_COMBOS {
                            compact_perm[i] = i as u16;
                        }
                        *perm = compact_perm;
                    }
                }
            }
        }
    }

    /// Pre-populate the showdown cache for all boards reachable in this subgame.
    /// Uses canonical cards to skip isomorphic boards.
    pub(crate) fn precompute_showdowns(&mut self) {
        let board = self.config.board;
        let n_board = board.count();

        if n_board >= 5 {
            // River: just one board
            self.sd_cache.get_or_compute(board, &self.combo_map);
        } else if n_board == 4 {
            // Turn: precompute only canonical river cards
            let canonical_rivers = canonical_next_cards(board);
            for cr in &canonical_rivers {
                let river_board = board.add(cr.card);
                self.sd_cache.get_or_compute(river_board, &self.combo_map);
            }
        } else if n_board == 3 {
            // Flop: precompute canonical turn × canonical river
            let canonical_turns = canonical_next_cards(board);
            for ct in &canonical_turns {
                let turn_board = board.add(ct.card);
                let canonical_rivers = canonical_next_cards(turn_board);
                for cr in &canonical_rivers {
                    let river_board = turn_board.add(cr.card);
                    self.sd_cache.get_or_compute(river_board, &self.combo_map);
                }
            }
        }
    }

    /// Compute initial reach probabilities in compact layout.
    /// reach[p][i] for i in 0..live_count holds the weight of live_to_combo[i].
    /// Indices live_count..NUM_COMBOS are zero.
    pub(crate) fn initial_reach(&self) -> [[f32; NUM_COMBOS]; 2] {
        let mut reach = [[0.0f32; NUM_COMBOS]; 2];
        let cm = &self.combo_map;
        for i in 0..cm.live_count {
            let orig = cm.live_to_combo[i] as usize;
            reach[0][i] = self.config.ranges[0].weights[orig];
            reach[1][i] = self.config.ranges[1].weights[orig];
        }
        reach
    }

    /// Recursively discover and create all tree nodes.
    /// Returns the tree node ID of the created node.
    pub(crate) fn build_tree(&mut self, state: &GameState, action_seq: &mut Vec<Action>) -> u32 {
        match state.node_type() {
            NodeType::Terminal(TerminalType::Fold(folder)) => {
                let id = self.tree.len() as u32;
                self.tree.push(TreeNode::TerminalFold(folder));
                id
            }
            NodeType::Terminal(TerminalType::Showdown) => {
                let id = self.tree.len() as u32;
                self.tree.push(TreeNode::TerminalShowdown);
                id
            }
            NodeType::Chance(next_street) => {
                // Depth-limited: if the next street matches the depth limit, insert NN leaf.
                if let Some(limit_street) = self.config.depth_limit {
                    if next_street == limit_street {
                        let id = self.tree.len() as u32;
                        #[cfg(feature = "nn")]
                        let vn_ptr = self.valuenet.as_ref()
                            .map(|vn| Arc::as_ptr(vn) as usize).unwrap_or(0);
                        #[cfg(not(feature = "nn"))]
                        let vn_ptr = 0usize;
                        self.tree.push(TreeNode::DepthLimitedLeaf { vn_ptr });
                        return id;
                    }
                }

                // Isomorphic card deduction: only build subtrees for canonical cards.
                // Isomorphic cards' contributions are recovered via combo permutations.
                let dead = state.board;
                let n_actual = (0..52u8).filter(|&c| !dead.contains(c)).count() as u8;

                let mut children = Vec::new();
                let mut iso_perms = Vec::new();
                if self.config.use_iso {
                    let canonical = canonical_next_cards(dead);
                    for cc in &canonical {
                        let next = state.deal_street(&[cc.card]);
                        let child_id = self.build_tree(&next, action_seq);
                        children.push((cc.card, child_id));
                        iso_perms.push(cc.perms.clone());
                    }
                } else {
                    // No ISO: enumerate all available cards individually
                    for card in 0..52u8 {
                        if dead.contains(card) { continue; }
                        let next = state.deal_street(&[card]);
                        let child_id = self.build_tree(&next, action_seq);
                        children.push((card, child_id));
                        iso_perms.push(Vec::new()); // no perms
                    }
                }
                let id = self.tree.len() as u32;
                self.tree.push(TreeNode::Chance { children, iso_perms, n_actual });
                id
            }
            NodeType::Decision(player) => {
                let actions = state.actions();
                let n_actions = actions.len();
                let cfr_idx = self.cfr.len() as u32;
                self.cfr.push(CfrData::new(n_actions, self.combo_map.live_count, 0));

                // Reserve slot (children filled after recursion)
                let id = self.tree.len() as u32;
                self.tree.push(TreeNode::Decision {
                    player,
                    actions: actions.clone(),
                    children: vec![0; n_actions],
                    cfr_idx,
                });

                self.node_index.insert(action_seq.clone(), id);

                let mut children = Vec::with_capacity(n_actions);
                for action in &actions {
                    let next = state.apply(*action);
                    action_seq.push(*action);
                    let child_id = self.build_tree(&next, action_seq);
                    action_seq.pop();
                    children.push(child_id);
                }

                if let TreeNode::Decision { children: ref mut ch, .. } = self.tree[id as usize] {
                    *ch = children;
                }

                id
            }
        }
    }

    // -----------------------------------------------------------------------
    // Public API (uses node_index for lookups)
    // -----------------------------------------------------------------------

    /// Get the average strategy at a node for combos with non-zero strategy weight.
    /// When cum_strategy is available, uses cumulative average. Otherwise falls back
    /// to regret-matched strategy from current regrets.
    /// Get the acting player at a given decision node.
    pub fn get_player(&self, action_seq: &[Action]) -> Option<Player> {
        let &tree_id = self.node_index.get(action_seq)?;
        match &self.tree[tree_id as usize] {
            TreeNode::Decision { player, .. } => Some(*player),
            _ => None,
        }
    }

    pub fn get_strategy(&self, action_seq: &[Action]) -> Option<Vec<(u16, Vec<(Action, f32)>)>> {
        let &tree_id = self.node_index.get(action_seq)?;
        let (actions, cfr_idx) = match &self.tree[tree_id as usize] {
            TreeNode::Decision { actions, cfr_idx, .. } => (actions, *cfr_idx as usize),
            _ => return None,
        };
        let cfr_node = &self.cfr[cfr_idx];
        let arena = self.arena.as_slice();
        let regrets = &arena[cfr_node.regret_offset..cfr_node.regret_offset + cfr_node.regret_len()];
        let cm = &self.combo_map;

        let mut result = Vec::new();
        for c in 0..cm.live_count {
            // Check if this combo has any meaningful strategy data.
            let has_data = if let Some(ref cum) = cfr_node.cum_strategy {
                let cum_total: f32 = (0..cfr_node.n_actions)
                    .map(|a| cum[c * cfr_node.n_actions + a])
                    .sum();
                cum_total > 0.0
            } else {
                // No cum_strategy: check if any positive regret exists (SoA layout).
                let lc = cfr_node.live_count;
                (0..cfr_node.n_actions).any(|a| regrets[a * lc + c] > 0)
            };
            if !has_data { continue; }

            let strat = cfr_node.average_strategy(c, arena, self.config.exploration_eps, self.config.softmax_temp);
            let action_probs: Vec<(Action, f32)> = actions.iter()
                .zip(strat.iter())
                .map(|(&a, &p)| (a, p))
                .collect();
            // Output original combo index for public API
            let orig = cm.live_to_combo[c];
            result.push((orig, action_probs));
        }
        Some(result)
    }

    /// Get average strategy for a specific combo at root node.
    /// `combo_idx` is the original 1326-combo index.
    pub fn root_strategy(&self, combo_idx: usize) -> Option<Vec<(Action, f32)>> {
        let (actions, cfr_idx) = match &self.tree[self.root_id as usize] {
            TreeNode::Decision { actions, cfr_idx, .. } => (actions, *cfr_idx as usize),
            _ => return None,
        };
        let compact = self.combo_map.combo_to_live[combo_idx];
        if compact < 0 { return None; }
        let strat = self.cfr[cfr_idx].average_strategy(compact as usize, self.arena.as_slice(), self.config.exploration_eps, self.config.softmax_temp);
        Some(actions.iter().zip(strat.iter()).map(|(&a, &p)| (a, p)).collect())
    }

    /// Iterate all decision node action sequences (for export).
    pub fn iter_decision_nodes(&self) -> impl Iterator<Item = &Vec<Action>> {
        self.node_index.keys()
    }

    /// Compute per-combo EV at root for the given player using average strategy.
    /// Returns array indexed by original combo index (0..1326).
    pub fn compute_ev(&self, player: Player) -> [f32; NUM_COMBOS] {
        let reach = self.initial_reach();
        let root_state = self.root_state();
        let arena = self.arena.as_slice();
        let cm = &self.combo_map;
        let raw_util = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, player, self.root_id,
        );

        let opp = 1 - player as usize;
        let mut per_card_opp = [0.0f32; 52];
        let mut total_opp_reach = 0.0f32;
        for oc in 0..cm.live_count {
            let r = reach[opp][oc];
            if r <= 0.0 { continue; }
            let orig = cm.live_to_combo[oc] as usize;
            let (o1, o2) = combo_from_index(orig as u16);
            total_opp_reach += r;
            per_card_opp[o1 as usize] += r;
            per_card_opp[o2 as usize] += r;
        }

        // Scatter compact results to original indices
        let mut ev = [0.0f32; NUM_COMBOS];
        for c in 0..cm.live_count {
            if reach[player as usize][c] <= 0.0 { continue; }
            let orig = cm.live_to_combo[c] as usize;
            let (c1, c2) = combo_from_index(orig as u16);
            let adj_opp = total_opp_reach - per_card_opp[c1 as usize] - per_card_opp[c2 as usize]
                + reach[opp][c];
            if adj_opp > 0.0 {
                ev[orig] = raw_util[c] / adj_opp;
            }
        }
        ev
    }

    /// Compute per-action EV at a specific decision node.
    /// Returns Vec of (Action, per-combo EV) for each action at the node (original index).
    /// `action_seq` identifies the path to the node.
    /// `player` is the player whose EV we compute (must be the acting player at this node).
    pub fn compute_action_evs(
        &self,
        action_seq: &[Action],
        player: Player,
    ) -> Option<Vec<(Action, [f32; NUM_COMBOS])>> {
        let &tree_id = self.node_index.get(action_seq)?;
        let (actions, children, _cfr_idx, acting) = match &self.tree[tree_id as usize] {
            TreeNode::Decision { player: p, actions, children, cfr_idx } => {
                (actions, children, *cfr_idx, *p)
            }
            _ => return None,
        };
        if acting != player { return None; }

        let arena = self.arena.as_slice();
        let cm = &self.combo_map;

        // Walk the path from root to compute reach at this node.
        let mut reach = self.initial_reach();
        let mut state = self.root_state();
        let mut current_id = self.root_id;

        for &action in action_seq {
            match &self.tree[current_id as usize] {
                TreeNode::Decision { player: act_p, actions: node_acts, children: node_ch, cfr_idx } => {
                    let p_idx = *act_p as usize;
                    let cfr_idx = *cfr_idx as usize;
                    let a_idx = match node_acts.iter().position(|&a| a == action) {
                        Some(i) => i,
                        None => return None,
                    };

                    // Update opponent's reach (not the eval player's)
                    if *act_p != player {
                        let avg = compute_avg_strat_flat(&self.cfr[cfr_idx], arena, self.config.exploration_eps, self.config.softmax_temp);
                        let n = node_acts.len();
                        for c in 0..cm.live_count {
                            reach[p_idx][c] *= avg[c * n + a_idx];
                        }
                    }

                    state = state.apply(action);
                    current_id = node_ch[a_idx];
                }
                _ => return None,
            }
        }

        // Compute opponent normalization at this node
        let opp = 1 - player as usize;
        let mut per_card_opp = [0.0f32; 52];
        let mut total_opp_reach = 0.0f32;
        for oc in 0..cm.live_count {
            let r = reach[opp][oc];
            if r <= 0.0 { continue; }
            let orig = cm.live_to_combo[oc] as usize;
            let (o1, o2) = combo_from_index(orig as u16);
            total_opp_reach += r;
            per_card_opp[o1 as usize] += r;
            per_card_opp[o2 as usize] += r;
        }

        // For each action, traverse child subtree and compute EV
        let mut result = Vec::new();
        for (a, &action) in actions.iter().enumerate() {
            let next_state = state.apply(action);
            let raw_util = eval_avg_strategy(
                &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
                &next_state, &reach, player, children[a],
            );

            // Normalize to per-combo EV, scatter to original indices
            let mut ev = [0.0f32; NUM_COMBOS];
            for c in 0..cm.live_count {
                if reach[player as usize][c] <= 0.0 { continue; }
                let orig = cm.live_to_combo[c] as usize;
                let (c1, c2) = combo_from_index(orig as u16);
                let adj_opp = total_opp_reach - per_card_opp[c1 as usize] - per_card_opp[c2 as usize]
                    + reach[opp][c];
                if adj_opp > 0.0 {
                    ev[orig] = raw_util[c] / adj_opp;
                }
            }
            result.push((action, ev));
        }
        Some(result)
    }

    /// Extract root OOP bet fractions and EV gaps per original combo index.
    /// Returns (bet_frac[1326], ev_gap[1326]) where gap = EV_bet - EV_check.
    pub fn root_bet_fractions(&self) -> Option<(Box<[f32; NUM_COMBOS]>, Box<[f32; NUM_COMBOS]>)> {
        let root_actions: Vec<Action> = vec![];
        let evs = self.compute_action_evs(&root_actions, OOP)?;
        if evs.len() < 2 { return None; }

        let arena = self.arena.as_slice();
        let cm = &self.combo_map;
        let cfr_idx = match &self.tree[self.root_id as usize] {
            TreeNode::Decision { cfr_idx, .. } => *cfr_idx as usize,
            _ => return None,
        };

        let bet_idx = evs.iter().position(|(a, _)| matches!(a, Action::Bet(_) | Action::AllIn))?;
        let chk_idx = evs.iter().position(|(a, _)| matches!(a, Action::Check))?;
        let bet_ev = &evs[bet_idx].1;
        let chk_ev = &evs[chk_idx].1;

        let mut bet_frac = Box::new([0.0f32; NUM_COMBOS]);
        let mut ev_gap = Box::new([0.0f32; NUM_COMBOS]);

        for c in 0..cm.live_count {
            let orig = cm.live_to_combo[c] as usize;
            let w = self.config.ranges[0].weights[orig];
            if w <= 0.0 { continue; }
            let avg = self.cfr[cfr_idx].average_strategy(c, arena, 0.0, self.config.softmax_temp);
            bet_frac[orig] = avg[bet_idx];
            ev_gap[orig] = bet_ev[orig] - chk_ev[orig];
        }
        Some((bet_frac, ev_gap))
    }

    /// Print EV analysis for the root node: per-combo bet EV, check EV, gap, and strategy.
    /// Returns (indifferent_count, total_count, avg_bet_pct_of_indifferent).
    pub fn root_ev_analysis(&self) -> (usize, usize, f32) {
        let root_actions: Vec<Action> = vec![];
        let evs = match self.compute_action_evs(&root_actions, OOP) {
            Some(v) => v,
            None => { println!("  No root OOP node found"); return (0, 0, 0.0); }
        };
        if evs.len() < 2 { println!("  Root has <2 actions"); return (0, 0, 0.0); }

        let arena = self.arena.as_slice();
        let cm = &self.combo_map;
        // Get root cfr_idx
        let cfr_idx = match &self.tree[self.root_id as usize] {
            TreeNode::Decision { cfr_idx, .. } => *cfr_idx as usize,
            _ => return (0, 0, 0.0),
        };

        // Find bet and check EVs by matching Action variants
        println!("  Root actions: {:?}", evs.iter().map(|(a, _)| format!("{}", a)).collect::<Vec<_>>());
        let bet_idx = evs.iter().position(|(a, _)| matches!(a, Action::Bet(_) | Action::AllIn));
        let chk_idx = evs.iter().position(|(a, _)| matches!(a, Action::Check));
        let (bet_idx, chk_idx) = match (bet_idx, chk_idx) {
            (Some(b), Some(c)) => (b, c),
            _ => { println!("  Root missing bet or check action"); return (0, 0, 0.0); }
        };
        let bet_ev = &evs[bet_idx].1;
        let chk_ev = &evs[chk_idx].1;

        struct ComboInfo {
            hand: String,
            bet_pct: f32,
            bet_ev: f32,
            chk_ev: f32,
            gap: f32,
        }

        let mut data: Vec<ComboInfo> = Vec::new();
        for c in 0..cm.live_count {
            let orig = cm.live_to_combo[c] as usize;
            let w = self.config.ranges[0].weights[orig];
            if w <= 0.0 { continue; }
            let (c1, c2) = combo_from_index(orig as u16);
            let hand = format!("{}{}", crate::card::card_to_string(c1), crate::card::card_to_string(c2));
            let avg = self.cfr[cfr_idx].average_strategy(c, arena, 0.0, self.config.softmax_temp);
            let bet_pct = avg[bet_idx] * 100.0;
            let bev = bet_ev[orig];
            let cev = chk_ev[orig];
            data.push(ComboInfo { hand, bet_pct, bet_ev: bev, chk_ev: cev, gap: bev - cev });
        }

        // Print key combos (AA, KK, QQ, JJ, TT, 99, 66, 87, A5d, A8d)
        println!("\n  === Key combo EV analysis ===");
        println!("  {:>6} | {:>6} | {:>8} | {:>8} | {:>8}", "Hand", "Bet%", "BetEV", "ChkEV", "Gap");
        let key_pairs: &[&str] = &["AA", "KK", "QQ", "JJ", "TT", "99", "66"];
        let key_specific: &[&str] = &["7d8d", "7h8h", "5dAd", "8dAd", "7s8s"];
        for d in &data {
            let h = &d.hand;
            let is_pair = h.len() == 4 && h.chars().nth(0) == h.chars().nth(2);
            let pair_name = if is_pair { format!("{}{}", h.chars().nth(0).unwrap(), h.chars().nth(2).unwrap()) } else { String::new() };
            if key_pairs.iter().any(|p| *p == pair_name) || key_specific.contains(&h.as_str()) {
                println!("  {:>6} | {:>5.1}% | {:>8.3} | {:>8.3} | {:>+8.4}", d.hand, d.bet_pct, d.bet_ev, d.chk_ev, d.gap);
            }
        }

        // Top 20 biggest EV gaps
        let mut sorted: Vec<&ComboInfo> = data.iter().collect();
        sorted.sort_by(|a, b| b.gap.abs().partial_cmp(&a.gap.abs()).unwrap());
        println!("\n  === Top 20 biggest EV gaps (root OOP) ===");
        println!("  {:>6} | {:>6} | {:>8} | {:>8} | {:>8}", "Hand", "Bet%", "BetEV", "ChkEV", "Gap");
        for d in sorted.iter().take(20) {
            println!("  {:>6} | {:>5.1}% | {:>8.3} | {:>8.3} | {:>+8.4}", d.hand, d.bet_pct, d.bet_ev, d.chk_ev, d.gap);
        }

        // Summary
        let indiff = data.iter().filter(|d| d.gap.abs() < 0.1).count();
        let total = data.len();
        let indiff_bet_avg: f32 = data.iter().filter(|d| d.gap.abs() < 0.1).map(|d| d.bet_pct).sum::<f32>() / indiff.max(1) as f32;
        println!("\n  Indifferent combos (|gap| < 0.1 chips): {}/{}", indiff, total);
        println!("  Avg bet% of indifferent combos: {:.1}%", indiff_bet_avg);
        let non_indiff_bet = data.iter().filter(|d| d.gap.abs() >= 0.1).map(|d| d.bet_pct).sum::<f32>() / data.iter().filter(|d| d.gap.abs() >= 0.1).count().max(1) as f32;
        println!("  Avg bet% of non-indifferent combos: {:.1}%", non_indiff_bet);
        (indiff, total, indiff_bet_avg)
    }

    /// Post-processing: smooth strategies for indifferent combos toward aggregate frequency.
    /// For each decision node, computes per-action EVs. Combos where the EV gap between
    /// best and worst played actions is less than `threshold` (in chips) have their strategy
    /// blended toward the node's aggregate frequency.
    /// `max_depth` limits smoothing to nodes with action sequence length ≤ max_depth.
    /// 0 = root only, 1 = root + first children, u32::MAX = all nodes.
    /// Returns the number of combos smoothed across all nodes.
    pub fn smooth_strategies(&mut self, threshold: f32) -> u32 {
        self.smooth_strategies_depth(threshold, u32::MAX)
    }

    /// Smooth strategies with depth limit on which nodes to process.
    pub fn smooth_strategies_depth(&mut self, threshold: f32, max_depth: u32) -> u32 {
        let action_seqs: Vec<Vec<Action>> = self.node_index.keys()
            .filter(|seq| seq.len() as u32 <= max_depth)
            .cloned().collect();
        let mut total_smoothed = 0u32;

        for action_seq in &action_seqs {
            let &tree_id = self.node_index.get(action_seq).unwrap();
            let (actions, cfr_idx, player) = match &self.tree[tree_id as usize] {
                TreeNode::Decision { player, actions, cfr_idx, .. } => {
                    (actions.clone(), *cfr_idx as usize, *player)
                }
                _ => continue,
            };

            let n_actions = actions.len();
            if n_actions <= 1 { continue; }

            // Get action EVs (indexed by original combo index)
            let action_evs = match self.compute_action_evs(action_seq, player) {
                Some(evs) => evs,
                None => continue,
            };

            let cm = &self.combo_map;
            let arena = self.arena.as_slice();
            let cfr_node = &self.cfr[cfr_idx];

            // Compute aggregate strategy (range-weighted average across all combos)
            let mut agg = vec![0.0f32; n_actions];
            let mut agg_total = 0.0f32;
            for c in 0..cm.live_count {
                let orig = cm.live_to_combo[c] as usize;
                let w = self.config.ranges[player as usize].weights[orig];
                if w <= 0.0 { continue; }
                let strat = cfr_node.average_strategy(c, arena, 0.0, self.config.softmax_temp);
                for a in 0..n_actions {
                    agg[a] += strat[a] * w;
                }
                agg_total += w;
            }
            if agg_total > 0.0 {
                for a in &mut agg { *a /= agg_total; }
            }

            // Collect per-combo smoothing decisions before mutating
            let mut updates: Vec<(usize, Vec<f32>)> = Vec::new();
            for c in 0..cm.live_count {
                let orig = cm.live_to_combo[c] as usize;
                let w = self.config.ranges[player as usize].weights[orig];
                if w <= 0.0 { continue; }

                let cfr_node = &self.cfr[cfr_idx];
                if cfr_node.cum_strategy.is_none() { break; }
                let cum = cfr_node.cum_strategy.as_ref().unwrap();

                // Current strategy
                let mut cum_total = 0.0f32;
                for a in 0..n_actions {
                    cum_total += cum[c * n_actions + a];
                }
                if cum_total <= 0.0 { continue; }
                let strat: Vec<f32> = (0..n_actions).map(|a| cum[c * n_actions + a] / cum_total).collect();

                // EV gap: only consider actions this combo actually plays (> 1%)
                let evs: Vec<f32> = action_evs.iter().map(|(_, ev)| ev[orig]).collect();
                let mut max_ev = f32::NEG_INFINITY;
                let mut min_ev = f32::INFINITY;
                for a in 0..n_actions {
                    if strat[a] > 0.01 {
                        max_ev = max_ev.max(evs[a]);
                        min_ev = min_ev.min(evs[a]);
                    }
                }
                if max_ev == f32::NEG_INFINITY || max_ev == min_ev && max_ev == 0.0 { continue; }
                let gap = max_ev - min_ev;

                if gap < threshold {
                    let blend = 1.0 - (gap / threshold);
                    let new_strat: Vec<f32> = (0..n_actions)
                        .map(|a| (1.0 - blend) * strat[a] + blend * agg[a])
                        .collect();
                    updates.push((c, new_strat.iter().map(|&s| s * cum_total).collect()));
                }
            }

            // Apply updates
            let smoothed_count = updates.len() as u32;
            if smoothed_count > 0 {
                let cfr_node = &mut self.cfr[cfr_idx];
                let cum = cfr_node.cum_strategy.as_mut().unwrap();
                for (c, new_cum) in updates {
                    for a in 0..n_actions {
                        cum[c * n_actions + a] = new_cum[a];
                    }
                }
                let node_desc = if action_seq.is_empty() {
                    "root".to_string()
                } else {
                    action_seq.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join(" → ")
                };
                println!("  Smoothed {} combos at node '{}' (threshold={:.2})", smoothed_count, node_desc, threshold);
                total_smoothed += smoothed_count;
            }
        }
        total_smoothed
    }

    /// Post-processing: purify strategies by zeroing actions below `min_pct`% and renormalizing.
    /// Applied to all decision nodes. Returns number of combos purified.
    pub fn purify(&mut self, min_pct: f32) -> u32 {
        let action_seqs: Vec<Vec<Action>> = self.node_index.keys().cloned().collect();
        let mut total = 0u32;

        for action_seq in &action_seqs {
            let &tree_id = self.node_index.get(action_seq).unwrap();
            let (n_actions, cfr_idx) = match &self.tree[tree_id as usize] {
                TreeNode::Decision { actions, cfr_idx, .. } => (actions.len(), *cfr_idx as usize),
                _ => continue,
            };
            if n_actions <= 1 { continue; }

            let cm = &self.combo_map;
            let cfr_node = &self.cfr[cfr_idx];
            if cfr_node.cum_strategy.is_none() { continue; }
            let cum = cfr_node.cum_strategy.as_ref().unwrap();
            let threshold = min_pct / 100.0;

            let mut updates: Vec<(usize, Vec<f32>)> = Vec::new();
            for c in 0..cm.live_count {
                let mut cum_total = 0.0f32;
                for a in 0..n_actions {
                    cum_total += cum[c * n_actions + a];
                }
                if cum_total <= 0.0 { continue; }

                let strat: Vec<f32> = (0..n_actions).map(|a| cum[c * n_actions + a] / cum_total).collect();

                // Check if any action is below threshold and non-zero
                let needs_purify = strat.iter().any(|&s| s > 0.0 && s < threshold);
                if !needs_purify { continue; }

                // Zero out low-frequency actions, renormalize
                let mut new_strat: Vec<f32> = strat.iter().map(|&s| if s < threshold { 0.0 } else { s }).collect();
                let new_total: f32 = new_strat.iter().sum();
                if new_total <= 0.0 { continue; }
                for s in &mut new_strat { *s /= new_total; }

                updates.push((c, new_strat.iter().map(|&s| s * cum_total).collect()));
            }

            if !updates.is_empty() {
                total += updates.len() as u32;
                let cfr_node = &mut self.cfr[cfr_idx];
                let cum = cfr_node.cum_strategy.as_mut().unwrap();
                for (c, new_cum) in updates {
                    for a in 0..n_actions {
                        cum[c * n_actions + a] = new_cum[a];
                    }
                }
            }
        }
        total
    }

    /// Post-processing: for indifferent combos (EV gap < `ev_threshold` chips),
    /// push strategy toward the most passive action (check > call > fold > bet/raise).
    /// `blend` controls how aggressively to push (0.0 = no change, 1.0 = pure passive).
    /// Returns number of combos adjusted.
    pub fn passive_tiebreak(&mut self, ev_threshold: f32, blend: f32) -> u32 {
        let action_seqs: Vec<Vec<Action>> = self.node_index.keys().cloned().collect();
        let mut total = 0u32;

        for action_seq in &action_seqs {
            let &tree_id = self.node_index.get(action_seq).unwrap();
            let (actions, cfr_idx, player) = match &self.tree[tree_id as usize] {
                TreeNode::Decision { player, actions, cfr_idx, .. } => {
                    (actions.clone(), *cfr_idx as usize, *player)
                }
                _ => continue,
            };

            let n_actions = actions.len();
            if n_actions <= 1 { continue; }

            let action_evs = match self.compute_action_evs(action_seq, player) {
                Some(evs) => evs,
                None => continue,
            };

            // Find the most passive action index (check > call > fold > bet/raise)
            let passive_idx = actions.iter().enumerate()
                .min_by_key(|(_, a)| match a {
                    Action::Check => 0,
                    Action::Call => 1,
                    Action::Fold => 2,
                    _ => 3, // bet/raise
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let cm = &self.combo_map;
            let cfr_node = &self.cfr[cfr_idx];
            if cfr_node.cum_strategy.is_none() { continue; }
            let cum = cfr_node.cum_strategy.as_ref().unwrap();

            let mut updates: Vec<(usize, Vec<f32>)> = Vec::new();
            for c in 0..cm.live_count {
                let orig = cm.live_to_combo[c] as usize;
                let w = self.config.ranges[player as usize].weights[orig];
                if w <= 0.0 { continue; }

                let mut cum_total = 0.0f32;
                for a in 0..n_actions {
                    cum_total += cum[c * n_actions + a];
                }
                if cum_total <= 0.0 { continue; }

                let strat: Vec<f32> = (0..n_actions).map(|a| cum[c * n_actions + a] / cum_total).collect();

                // Is this combo mixed? (plays at least 2 actions above 1%)
                let active_actions = strat.iter().filter(|&&s| s > 0.01).count();
                if active_actions < 2 { continue; }

                // Compute EV gap across played actions
                let evs: Vec<f32> = action_evs.iter().map(|(_, ev)| ev[orig]).collect();
                let mut max_ev = f32::NEG_INFINITY;
                let mut min_ev = f32::INFINITY;
                for a in 0..n_actions {
                    if strat[a] > 0.01 {
                        max_ev = max_ev.max(evs[a]);
                        min_ev = min_ev.min(evs[a]);
                    }
                }
                let gap = max_ev - min_ev;
                if gap >= ev_threshold { continue; }

                // Blend toward passive action
                let strength = blend * (1.0 - gap / ev_threshold);
                let mut new_strat = strat.clone();
                let passive_mass: f32 = (0..n_actions)
                    .filter(|&a| a != passive_idx)
                    .map(|a| new_strat[a] * strength)
                    .sum();
                for a in 0..n_actions {
                    if a == passive_idx {
                        new_strat[a] += passive_mass;
                    } else {
                        new_strat[a] *= 1.0 - strength;
                    }
                }

                updates.push((c, new_strat.iter().map(|&s| s * cum_total).collect()));
            }

            if !updates.is_empty() {
                total += updates.len() as u32;
                let cfr_node = &mut self.cfr[cfr_idx];
                let cum = cfr_node.cum_strategy.as_mut().unwrap();
                for (c, new_cum) in updates {
                    for a in 0..n_actions {
                        cum[c * n_actions + a] = new_cum[a];
                    }
                }
            }
        }
        total
    }

    /// Compute per-combo best response EV for `br_player`.
    /// Returns array indexed by original combo index (0..1326).
    pub fn best_response_ev(&self, br_player: Player) -> [f32; NUM_COMBOS] {
        let reach = self.initial_reach();
        let root_state = self.root_state();
        let arena = self.arena.as_slice();
        let cm = &self.combo_map;
        let raw_util = br_traverse(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, br_player, self.root_id,
        );

        let opp = 1 - br_player as usize;
        let mut per_card_opp = [0.0f32; 52];
        let mut total_opp_reach = 0.0f32;
        for oc in 0..cm.live_count {
            let r = reach[opp][oc];
            if r <= 0.0 { continue; }
            let orig = cm.live_to_combo[oc] as usize;
            let (o1, o2) = combo_from_index(orig as u16);
            total_opp_reach += r;
            per_card_opp[o1 as usize] += r;
            per_card_opp[o2 as usize] += r;
        }

        let mut ev = [0.0f32; NUM_COMBOS];
        for c in 0..cm.live_count {
            if reach[br_player as usize][c] <= 0.0 { continue; }
            let orig = cm.live_to_combo[c] as usize;
            let (c1, c2) = combo_from_index(orig as u16);
            let adj_opp = total_opp_reach - per_card_opp[c1 as usize] - per_card_opp[c2 as usize]
                + reach[opp][c];
            if adj_opp > 0.0 {
                ev[orig] = raw_util[c] / adj_opp;
            }
        }
        ev
    }

    /// Diagnostic: verify constant-sum property and print detailed exploitability breakdown.
    pub fn exploitability_diagnostic(&self) {
        let reach = self.initial_reach();
        let root_state = self.root_state();
        let arena = self.arena.as_slice();
        let cm = &self.combo_map;

        // Eval with average strategy for both players
        let raw_eval_oop = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, OOP, self.root_id,
        );
        let raw_eval_ip = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, IP, self.root_id,
        );

        // BR for both players
        let raw_br_oop = br_traverse(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, OOP, self.root_id,
        );
        let raw_br_ip = br_traverse(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, IP, self.root_id,
        );

        let mut v_eval_oop = 0.0f64;
        let mut v_eval_ip = 0.0f64;
        let mut v_br_oop = 0.0f64;
        let mut v_br_ip = 0.0f64;
        for c in 0..cm.live_count {
            let orig = cm.live_to_combo[c] as usize;
            let w0 = self.config.ranges[0].weights[orig] as f64;
            if w0 > 0.0 {
                v_eval_oop += w0 * raw_eval_oop[c] as f64;
                v_br_oop += w0 * raw_br_oop[c] as f64;
            }
            let w1 = self.config.ranges[1].weights[orig] as f64;
            if w1 > 0.0 {
                v_eval_ip += w1 * raw_eval_ip[c] as f64;
                v_br_ip += w1 * raw_br_ip[c] as f64;
            }
        }

        let pot = self.config.pot as f64;
        let v_eval_sum = v_eval_oop + v_eval_ip;
        let v_br_sum = v_br_oop + v_br_ip;

        // Z from initial board (for reference, but NOT used for exploitability)
        let z_board = compute_z(&self.config.ranges[0], &self.config.ranges[1], self.config.board);

        println!("  --- Exploitability Diagnostic ---");
        println!("  V_eval_oop = {:.4}, V_eval_ip = {:.4}", v_eval_oop, v_eval_ip);
        println!("  V_eval_sum = {:.4}", v_eval_sum);
        println!("  Z_board*pot= {:.4} (ref only, not used)", z_board * pot);
        println!("  V_br_oop   = {:.4}, V_br_ip = {:.4}", v_br_oop, v_br_ip);
        println!("  V_br_sum   = {:.4}", v_br_sum);
        // Per-player exploitability (approximate: uses V_eval_sum/pot as Z for both)
        let z_eff = v_eval_sum / pot;
        println!("  Expl_oop   = {:.4}%", (v_br_oop - v_eval_oop) / z_eff / pot * 100.0);
        println!("  Expl_ip    = {:.4}%", (v_br_ip - v_eval_ip) / z_eff / pot * 100.0);
        // Total exploitability (exact formula)
        let expl = (v_br_sum - v_eval_sum) * pot / (2.0 * v_eval_sum);
        println!("  Expl_total = {:.4}%", expl / pot * 100.0);
    }

    /// Compute exploitability using eval-normalized formula.
    ///
    /// In multi-street games, the chance node's 1/N averaging creates a
    /// strategy-dependent normalization factor: fold terminals (before chance)
    /// carry weight 1 per pair, while river terminals carry weight ~44/48.
    /// No fixed Z satisfies V_eval_sum = Z * pot for all strategies.
    ///
    /// Instead, we compute V_eval_sum directly (which IS constant-sum within
    /// each strategy profile) and use it as the normalizer:
    ///   expl = (V_br_sum - V_eval_sum) * pot / (2 * V_eval_sum)
    ///
    /// At Nash: V_br = V_eval for each player, so expl = 0.
    /// For river-only (no chance node): this equals the standard Z-based formula.
    pub fn exploitability(&self) -> f32 {
        let reach = self.initial_reach();
        let root_state = self.root_state();
        let arena = self.arena.as_slice();
        let cm = &self.combo_map;

        // Eval with average strategy for both players
        let raw_eval_oop = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, OOP, self.root_id,
        );
        let raw_eval_ip = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, IP, self.root_id,
        );

        // BR for both players
        let raw_br_oop = br_traverse(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, OOP, self.root_id,
        );
        let raw_br_ip = br_traverse(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, IP, self.root_id,
        );

        // Weighted sums
        let mut v_eval_sum = 0.0f64;
        let mut v_br_sum = 0.0f64;
        for c in 0..cm.live_count {
            let orig = cm.live_to_combo[c] as usize;
            let w0 = self.config.ranges[0].weights[orig] as f64;
            if w0 > 0.0 {
                v_eval_sum += w0 * raw_eval_oop[c] as f64;
                v_br_sum += w0 * raw_br_oop[c] as f64;
            }
            let w1 = self.config.ranges[1].weights[orig] as f64;
            if w1 > 0.0 {
                v_eval_sum += w1 * raw_eval_ip[c] as f64;
                v_br_sum += w1 * raw_br_ip[c] as f64;
            }
        }

        if v_eval_sum.abs() <= 1e-10 { return 0.0; }

        // expl = (V_br_sum - V_eval_sum) * pot / (2 * V_eval_sum)
        let pot = self.config.pot as f64;
        ((v_br_sum - v_eval_sum) * pot / (2.0 * v_eval_sum)) as f32
    }

    /// Overall EV for each player (properly normalized with card removal).
    /// Returns (oop_ev, ip_ev). In a zero-sum no-rake game, oop_ev + ip_ev ≈ pot.
    /// Uses the same raw counterfactual values as exploitability(), split per player.
    pub fn overall_ev(&self) -> (f32, f32) {
        let reach = self.initial_reach();
        let root_state = self.root_state();
        let arena = self.arena.as_slice();
        let cm = &self.combo_map;

        let raw_oop = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, OOP, self.root_id,
        );
        let raw_ip = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, IP, self.root_id,
        );

        let mut v_oop = 0.0f64;
        let mut v_ip = 0.0f64;
        for c in 0..cm.live_count {
            let orig = cm.live_to_combo[c] as usize;
            let w0 = self.config.ranges[0].weights[orig] as f64;
            if w0 > 0.0 {
                v_oop += w0 * raw_oop[c] as f64;
            }
            let w1 = self.config.ranges[1].weights[orig] as f64;
            if w1 > 0.0 {
                v_ip += w1 * raw_ip[c] as f64;
            }
        }

        // Z = v_eval_sum / pot; per-player EV = v_player / Z
        let v_sum = v_oop + v_ip;
        let pot = self.config.pot as f64;
        if v_sum.abs() <= 1e-10 || pot <= 0.0 {
            return (0.0, 0.0);
        }
        let z = v_sum / pot;
        ((v_oop / z) as f32, (v_ip / z) as f32)
    }

    /// Return the live_to_combo mapping: compact_idx -> original combo index.
    pub fn live_to_combo(&self) -> &[u16] {
        &self.combo_map.live_to_combo[..self.combo_map.live_count]
    }

    /// Return combo_to_live mapping: original combo index -> compact idx (or -1).
    pub fn combo_to_live(&self) -> &[i16; NUM_COMBOS] {
        &self.combo_map.combo_to_live
    }

    /// Compute per-action continuation EV at root for each combo (compact index).
    /// Returns Vec<(Action, Vec<f32>)> where Vec<f32>[c] = raw counterfactual utility
    /// for compact combo c when taking that action, with subtree strategy fixed.
    /// Used for 2-stage equilibrium selection (root-only re-optimization).
    pub fn root_action_evs(&self) -> Vec<(Action, Vec<f32>)> {
        let (actions, _cfr_idx) = match &self.tree[self.root_id as usize] {
            TreeNode::Decision { actions, cfr_idx, .. } => (actions.clone(), *cfr_idx),
            _ => return Vec::new(),
        };
        let children = match &self.tree[self.root_id as usize] {
            TreeNode::Decision { children, .. } => children.clone(),
            _ => return Vec::new(),
        };
        let reach = self.initial_reach();
        let root_state = self.root_state();
        let arena = self.arena.as_slice();
        let cm = &self.combo_map;

        let mut result = Vec::new();
        for (a, &action) in actions.iter().enumerate() {
            let next_state = root_state.apply(action);
            let action_util = eval_avg_strategy(
                &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
                &next_state, &reach, OOP, children[a],
            );
            result.push((action, action_util[..cm.live_count].to_vec()));
        }
        result
    }

    /// Set root cum_strategy for a specific compact combo index.
    /// `action_probs` maps action index to probability.
    /// Overwrites cum_strategy so average_strategy returns these exact probs.
    pub fn set_root_mix(&mut self, compact_idx: usize, action_probs: &[f32]) {
        let cfr_idx = match &self.tree[self.root_id as usize] {
            TreeNode::Decision { cfr_idx, .. } => *cfr_idx as usize,
            _ => return,
        };
        let n = self.cfr[cfr_idx].n_actions;
        assert_eq!(action_probs.len(), n);
        self.cfr[cfr_idx].ensure_cum_strategy();
        if let Some(ref mut cum) = self.cfr[cfr_idx].cum_strategy {
            // Set cum_strategy so that normalization yields the desired probs.
            // Use a large total to dominate any prior accumulation.
            let scale = 1e6;
            for a in 0..n {
                cum[compact_idx * n + a] = action_probs[a] * scale;
            }
        }
    }

    /// Exploitability as percentage of pot (clamped to ≥ 0).
    pub fn exploitability_pct(&self) -> f32 {
        let expl = self.exploitability().max(0.0);
        if self.config.pot > 0 { expl / self.config.pot as f32 * 100.0 } else { 0.0 }
    }

    /// Exploitability of the current (regret-matched) strategy, ignoring cum_strategy.
    /// Uses thread-local flag to force compute_avg_strat_flat to use regrets.
    pub fn exploitability_current(&self) -> f32 {
        FORCE_CURRENT_STRATEGY.with(|f| f.set(true));
        let expl = self.exploitability();
        FORCE_CURRENT_STRATEGY.with(|f| f.set(false));
        expl
    }

    /// Current strategy exploitability as percentage of pot (clamped to ≥ 0).
    pub fn exploitability_current_pct(&self) -> f32 {
        let expl = self.exploitability_current().max(0.0);
        if self.config.pot > 0 { expl / self.config.pot as f32 * 100.0 } else { 0.0 }
    }

    /// Regret and policy diagnostics for convergence analysis.
    /// Returns: (avg_pos_regret_norm, near_zero_frac, policy_l1_drift)
    /// - avg_pos_regret_norm: mean L1 norm of positive regrets per infoset-combo
    /// - near_zero_frac: fraction of infoset-combos with max positive regret < 0.01
    /// - policy_l1_drift: L1 distance of current root strategy from `prev_root_strat` (if provided)
    pub fn regret_policy_stats(&self, prev_root_strat: Option<&[f32]>) -> (f32, f32, f32) {
        let arena = self.arena.as_slice();
        let mut total_pos_norm = 0.0f64;
        let mut near_zero_count = 0u64;
        let mut total_entries = 0u64;

        for cfr_node in &self.cfr {
            let n = cfr_node.n_actions;
            let lc = cfr_node.live_count;
            let regrets = &arena[cfr_node.regret_offset..cfr_node.regret_offset + cfr_node.regret_len()];
            let scale = cfr_node.regret_scale.get();

            for c in 0..lc {
                let mut pos_sum = 0.0f32;
                let mut max_pos = 0.0f32;
                for a in 0..n {
                    let r = (regrets[a * lc + c] as f32 * scale).max(0.0);
                    pos_sum += r;
                    if r > max_pos { max_pos = r; }
                }
                total_pos_norm += pos_sum as f64;
                if max_pos < 0.01 { near_zero_count += 1; }
                total_entries += 1;
            }
        }

        let avg_norm = if total_entries > 0 { (total_pos_norm / total_entries as f64) as f32 } else { 0.0 };
        let nz_frac = if total_entries > 0 { near_zero_count as f32 / total_entries as f32 } else { 0.0 };

        // Root policy L1 drift
        let drift = if let Some(prev) = prev_root_strat {
            let root_cfr_idx = match &self.tree[self.root_id as usize] {
                TreeNode::Decision { cfr_idx, .. } => *cfr_idx as usize,
                _ => return (avg_norm, nz_frac, 0.0),
            };
            let cfr_node = &self.cfr[root_cfr_idx];
            let n = cfr_node.n_actions;
            let lc = cfr_node.live_count;
            let cur = compute_avg_strat_flat(cfr_node, arena, self.config.exploration_eps, self.config.softmax_temp);
            if prev.len() == cur.len() {
                let mut d = 0.0f32;
                for i in 0..cur.len() {
                    d += (cur[i] - prev[i]).abs();
                }
                d / lc as f32 // avg L1 per combo
            } else { 0.0 }
        } else { 0.0 };

        (avg_norm, nz_frac, drift)
    }

    /// Get flat root strategy snapshot (for drift calculation between iterations).
    pub fn root_strat_snapshot(&self) -> Vec<f32> {
        let root_cfr_idx = match &self.tree[self.root_id as usize] {
            TreeNode::Decision { cfr_idx, .. } => *cfr_idx as usize,
            _ => return Vec::new(),
        };
        let cfr_node = &self.cfr[root_cfr_idx];
        compute_avg_strat_flat(cfr_node, self.arena.as_slice(), self.config.exploration_eps, self.config.softmax_temp)
    }

    /// Per-combo EVs at the root for both players (original combo index space).
    /// Used for NN training data generation.
    pub fn root_ev_per_combo(&self) -> [[f32; NUM_COMBOS]; 2] {
        let reach = self.initial_reach();
        let root_state = self.root_state();
        let arena = self.arena.as_slice();
        let cm = &self.combo_map;

        let raw_oop = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, OOP, self.root_id,
        );
        let raw_ip = eval_avg_strategy(
            &self.tree, &self.cfr, arena, &self.config, &self.sd_cache, cm,
            &root_state, &reach, IP, self.root_id,
        );

        // Map compact indices back to original combo indices
        let mut ev = [[0.0f32; NUM_COMBOS]; 2];
        for i in 0..cm.live_count {
            let orig = cm.live_to_combo[i] as usize;
            ev[0][orig] = raw_oop[i];
            ev[1][orig] = raw_ip[i];
        }
        ev
    }

    /// Return per-node regret-matched strategy for convergence measurement.
    /// Each node returns a flat Vec<f32> of size NUM_COMBOS * n_actions.
    /// Used for pruning potential analysis (comparing strategies between checkpoints).
    pub fn strategy_snapshot(&self) -> Vec<Vec<f32>> {
        let arena = self.arena.as_slice();
        self.cfr.iter().map(|node| {
            let n = node.n_actions;
            let lc = node.live_count;
            let regrets = &arena[node.regret_offset..node.regret_offset + node.regret_len()];
            let scale = node.regret_scale.get();
            let mut strats = vec![0.0f32; lc * n];
            for c in 0..lc {
                let out_base = c * n;
                let mut pos_sum = 0.0f32;
                for a in 0..n {
                    let r = (regrets[a * lc + c] as f32 * scale).max(0.0);
                    strats[out_base + a] = r;
                    pos_sum += r;
                }
                if pos_sum > 0.0 {
                    let inv = 1.0 / pos_sum;
                    for a in 0..n { strats[out_base + a] *= inv; }
                } else {
                    let u = 1.0 / n as f32;
                    for a in 0..n { strats[out_base + a] = u; }
                }
            }
            strats
        }).collect()
    }

    /// Return number of actions per CfrData node (for analysis).
    pub fn node_action_counts(&self) -> Vec<usize> {
        self.cfr.iter().map(|node| node.n_actions).collect()
    }

    /// Detailed memory breakdown in bytes.
    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        // CfrData
        let regrets_bytes = self.arena.len * std::mem::size_of::<ArenaInt>();
        let mut cum_strategy_bytes: usize = 0;
        let mut cfr_overhead: usize = 0;
        let mut action_distribution = [0usize; 12]; // index = n_actions, count
        for data in &self.cfr {
            if let Some(ref cs) = data.cum_strategy {
                cum_strategy_bytes += cs.len() * std::mem::size_of::<f32>()
                    + std::mem::size_of::<Vec<f32>>();
            }
            cfr_overhead += std::mem::size_of::<CfrData>();
            let na = data.n_actions.min(action_distribution.len() - 1);
            action_distribution[na] += 1;
        }

        // TreeNode
        let mut tree_bytes: usize = 0;
        let mut decision_count = 0usize;
        let mut chance_count = 0usize;
        let mut terminal_count = 0usize;
        let mut iso_perm_bytes: usize = 0;
        for node in &self.tree {
            tree_bytes += std::mem::size_of::<TreeNode>(); // enum discriminant + inline data
            match node {
                TreeNode::Decision { actions, children, .. } => {
                    decision_count += 1;
                    tree_bytes += actions.capacity() * std::mem::size_of::<Action>();
                    tree_bytes += children.capacity() * std::mem::size_of::<u32>();
                }
                TreeNode::Chance { children, iso_perms, .. } => {
                    chance_count += 1;
                    tree_bytes += children.capacity() * std::mem::size_of::<(u8, u32)>();
                    for perms in iso_perms {
                        iso_perm_bytes += perms.len() * std::mem::size_of::<[u16; NUM_COMBOS]>();
                        iso_perm_bytes += std::mem::size_of::<Vec<[u16; NUM_COMBOS]>>();
                    }
                    tree_bytes += iso_perms.capacity() * std::mem::size_of::<Vec<[u16; NUM_COMBOS]>>();
                }
                TreeNode::TerminalFold(_) | TreeNode::TerminalShowdown
                | TreeNode::DepthLimitedLeaf { .. } => {
                    terminal_count += 1;
                }
            }
        }
        // Vec<TreeNode> overhead
        tree_bytes += self.tree.capacity() * std::mem::size_of::<TreeNode>();

        // node_index
        // FxHashMap<Vec<Action>, u32>: each entry has a Vec (24 bytes + heap) + u32 + hash overhead
        let mut node_index_bytes: usize = 0;
        for (key, _) in &self.node_index {
            node_index_bytes += std::mem::size_of::<Vec<Action>>()
                + key.capacity() * std::mem::size_of::<Action>()
                + std::mem::size_of::<u32>()
                + 8; // hash table overhead estimate per entry
        }

        // Showdown cache
        let mut sd_cache_bytes: usize = 0;
        let mut sd_entries = 0usize;
        for (_, sd) in &self.sd_cache.entries {
            sd_entries += 1;
            sd_cache_bytes += sd.sorted.capacity()
                * std::mem::size_of::<(u16, Strength, Card, Card)>();
            sd_cache_bytes += std::mem::size_of::<Vec<(u16, Strength, Card, Card)>>();
        }

        MemoryBreakdown {
            regrets_bytes,
            cum_strategy_bytes,
            cfr_overhead,
            tree_bytes,
            iso_perm_bytes,
            node_index_bytes,
            sd_cache_bytes,
            sd_entries,
            decision_count,
            chance_count,
            terminal_count,
            action_distribution,
            live_count: self.combo_map.live_count,
        }
    }
}

/// Detailed memory usage breakdown.
pub struct MemoryBreakdown {
    pub regrets_bytes: usize,
    pub cum_strategy_bytes: usize,
    pub cfr_overhead: usize,
    pub tree_bytes: usize,
    pub iso_perm_bytes: usize,
    pub node_index_bytes: usize,
    pub sd_cache_bytes: usize,
    pub sd_entries: usize,
    pub decision_count: usize,
    pub chance_count: usize,
    pub terminal_count: usize,
    pub action_distribution: [usize; 12],
    pub live_count: usize,
}

impl MemoryBreakdown {
    pub fn total_bytes(&self) -> usize {
        self.regrets_bytes + self.cum_strategy_bytes + self.cfr_overhead
            + self.tree_bytes + self.iso_perm_bytes
            + self.node_index_bytes + self.sd_cache_bytes
    }

    pub fn print(&self) {
        let mb = |b: usize| b as f64 / 1024.0 / 1024.0;
        let gb = |b: usize| b as f64 / 1024.0 / 1024.0 / 1024.0;
        println!("  CfrData regrets:   {:>10.1} MB", mb(self.regrets_bytes));
        println!("  CfrData cum_strat: {:>10.1} MB", mb(self.cum_strategy_bytes));
        println!("  CfrData overhead:  {:>10.1} MB", mb(self.cfr_overhead));
        println!("  Tree nodes:        {:>10.1} MB", mb(self.tree_bytes));
        println!("  Iso permutations:  {:>10.1} MB", mb(self.iso_perm_bytes));
        println!("  Node index:        {:>10.1} MB", mb(self.node_index_bytes));
        println!("  Showdown cache:    {:>10.1} MB ({} boards)", mb(self.sd_cache_bytes), self.sd_entries);
        println!("  ---");
        println!("  Total estimated:   {:>10.1} MB ({:.2} GB)", mb(self.total_bytes()), gb(self.total_bytes()));
        println!();
        println!("  Nodes: {} decision, {} chance, {} terminal, {} total",
                 self.decision_count, self.chance_count, self.terminal_count,
                 self.decision_count + self.chance_count + self.terminal_count);
        println!();
        println!("  Action distribution:");
        for (n, &count) in self.action_distribution.iter().enumerate() {
            if count > 0 {
                let slot_bytes = self.live_count * n * std::mem::size_of::<ArenaInt>();
                println!("    {}-action: {:>8} nodes ({:>6.1} MB regrets, {:>5} B/node)",
                         n, count, count as f64 * slot_bytes as f64 / 1024.0 / 1024.0, slot_bytes);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Joint normalization constant Z
// ---------------------------------------------------------------------------

/// Compute Z = Σ_{c_oop, c_ip: compatible with board} w_oop[c_oop] * w_ip[c_ip]
fn compute_z(range_oop: &Range, range_ip: &Range, board: Hand) -> f64 {
    let mut total_ip = 0.0f64;
    let mut per_card_ip = [0.0f64; 52];
    for c in 0..NUM_COMBOS {
        let w = range_ip.weights[c] as f64;
        if w <= 0.0 { continue; }
        let (c1, c2) = combo_from_index(c as u16);
        if board.contains(c1) || board.contains(c2) { continue; }
        total_ip += w;
        per_card_ip[c1 as usize] += w;
        per_card_ip[c2 as usize] += w;
    }

    let mut z = 0.0f64;
    for c in 0..NUM_COMBOS {
        let w0 = range_oop.weights[c] as f64;
        if w0 <= 0.0 { continue; }
        let (c1, c2) = combo_from_index(c as u16);
        if board.contains(c1) || board.contains(c2) { continue; }
        let adj_ip = total_ip - per_card_ip[c1 as usize] - per_card_ip[c2 as usize]
            + range_ip.weights[c] as f64;
        z += w0 * adj_ip;
    }
    z
}

// ---------------------------------------------------------------------------
// DCFR discounting (free function operating on CfrData)
// ---------------------------------------------------------------------------

fn discount_all(cfr: &mut [CfrData], t: u32, mode: &DcfrMode, cfr_plus: bool, arena_ptr: *mut ArenaInt) {
    let t_f = t as f64;

    let (alpha, beta, gamma) = match mode {
        DcfrMode::Standard => (1.5, 0.0, 2.0),
        DcfrMode::HsSchedule => (
            1.0 + 3.0 * t_f / 1000.0,  // alpha(t) = 1 + 3t/1000
            -1.0 - t_f / 500.0,          // beta(t) = -1 - t/500
            30.0 - 5.0 * t_f / 1000.0,   // gamma(t) = 30 - 5t/1000
        ),
        DcfrMode::BinaryTrick => (1.5, 0.0, 3.0),
        DcfrMode::Linear => (1.0, 0.0, 1.0),
        DcfrMode::Custom(a, b, g) => (*a, *b, *g),
    };

    let alpha_d = (t_f.powf(alpha) / (t_f.powf(alpha) + 1.0)) as f32;
    let gamma_d = ((t_f / (t_f + 1.0)).powf(gamma)) as f32;

    // HS-DCFR needs beta discounting for negative regrets.
    // This requires element-level decode/encode because i16 scale can't
    // differentiate positive vs negative. O(elements) but HS-DCFR should
    // need far fewer iterations to compensate.
    if !cfr_plus && *mode == DcfrMode::HsSchedule {
        let beta_d = (t_f.powf(beta) / (t_f.powf(beta) + 1.0)) as f32;
        let ptr_usize = arena_ptr as usize; // for Send across par_iter
        // Element-level discount: decode, apply alpha_d/beta_d, re-encode
        cfr.par_iter().for_each(|node| {
            let n = node.live_count * node.n_actions;
            if n == 0 { return; }
            let offset = node.regret_offset;
            let scale = node.regret_scale.get();
            if scale == 0.0 { return; }
            let ap = ptr_usize as *mut ArenaInt;
            let i16_slice = unsafe { std::slice::from_raw_parts_mut(ap.add(offset), n) };
            // Find new max for re-encoding
            let mut max_abs = 0.0f32;
            for v in i16_slice.iter() {
                let f = *v as f32 * scale;
                let d = if f >= 0.0 { f * alpha_d } else { f * beta_d };
                if d.abs() > max_abs { max_abs = d.abs(); }
            }
            let new_scale = if max_abs > 0.0 { max_abs / 32767.0 } else { 0.0 };
            let inv_scale = if new_scale > 0.0 { 1.0 / new_scale } else { 0.0 };
            for v in i16_slice.iter_mut() {
                let f = *v as f32 * scale;
                let d = if f >= 0.0 { f * alpha_d } else { f * beta_d };
                *v = (d * inv_scale).round().clamp(-ARENA_MAX, ARENA_MAX) as ArenaInt;
            }
            node.regret_scale.set(new_scale);
        });
    } else {
        // Standard / BinaryTrick / HS with CFR+ (no negative regrets):
        // Just multiply scale factor. O(nodes).
        cfr.par_iter_mut().for_each(|node| {
            node.regret_scale.set(node.regret_scale.get() * alpha_d);
        });
    }

    // Cum strategy discount
    cfr.par_iter_mut().for_each(|node| {
        if let Some(ref mut cum) = node.cum_strategy {
            for s in cum.iter_mut() { *s *= gamma_d; }
        }
    });
}

// ---------------------------------------------------------------------------
// Depth-limited leaf evaluation via neural network
// ---------------------------------------------------------------------------

/// Evaluate a depth-limited leaf node using the neural network.
///
/// Enumerates ALL possible next cards (not canonical — no isomorphism needed
/// since NN handles raw boards). For each card, converts compact reach to
/// Range format, zeros card-blocked combos, calls NN, and maps output back
/// to compact indices.
///
/// Equity-based depth-limited leaf: computes exact river checkdown value.
/// For each river card, compute showdown utility and average.
/// This serves as both a fallback (no NN) and a plumbing verification tool.
#[cfg(feature = "nn")]
fn depth_limited_equity(
    combo_map: &ComboMap,
    config: &SubgameConfig,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    traverser: Player,
    out: &mut [f32; NUM_COMBOS],
) {
    let live_count = combo_map.live_count;
    for v in out[..live_count].iter_mut() { *v = 0.0; }

    let board = state.board;
    let available_cards: Vec<u8> = (0..52u8).filter(|&c| !board.contains(c)).collect();
    let n_cards = available_cards.len();
    if n_cards == 0 { return; }

    let mut card_out = [0.0f32; NUM_COMBOS];
    for &card in &available_cards {
        let river_board = board.add(card);
        let bsd = BoardShowdown::compute(river_board, combo_map);
        for v in card_out.iter_mut() { *v = 0.0; }
        showdown_utility_with_precomputed(config, &bsd, state, reach, traverser, &mut card_out);
        for i in 0..live_count {
            out[i] += card_out[i];
        }
    }
    let inv = 1.0 / n_cards as f32;
    for i in 0..live_count {
        out[i] *= inv;
    }

}

/// Equity-anchored NN depth-limited leaf values.
///
/// The NN predicts per-combo counterfactual values, but has a systematic positive
/// mean bias. After denormalization by pot × opp_reach, this bias gets amplified
/// by pot size — bigger pots = bigger bias — creating a preference for actions
/// that grow the pot (bet/raise/all-in degeneracy).
///
/// Fix: For each dealt card, compute both NN and equity leaf values, then shift
/// NN values so their reach-weighted mean matches equity's mean. This:
/// - Preserves NN's relative combo ordering (strategic nuance: semi-bluffs, etc.)
/// - Anchors the absolute level to equity (no pot-scaling bias)
/// - Eliminates the all-in degeneracy caused by positive mean inflation
///
/// Modes (NN_CALIBRATE env var):
///   "anchor" (default) — equity-anchored: NN ranking + equity mean level
///   "blend"            — legacy linear blend (NN_BLEND controls mix ratio)
///   "off"              — pure NN (no calibration, will likely produce all-in)
#[cfg(feature = "nn")]
fn depth_limited_utility(
    vn_ptr: usize,
    config: &SubgameConfig,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    traverser: Player,
    out: &mut [f32; NUM_COMBOS],
) {
    use crate::card::combo_from_index;

    let live_count = combo_map.live_count;
    for v in out[..live_count].iter_mut() { *v = 0.0; }

    if vn_ptr == 0 {
        // No valuenet: compute equity-based leaf (exact river checkdown value)
        depth_limited_equity(combo_map, config, state, reach, traverser, out);
        return;
    }

    // Calibration mode
    use std::sync::OnceLock;
    static CALIBRATE_MODE: OnceLock<String> = OnceLock::new();
    let calibrate_mode = CALIBRATE_MODE.get_or_init(|| {
        std::env::var("NN_CALIBRATE").unwrap_or_else(|_| "anchor".to_string())
    });

    // Legacy blend factor (only used in "blend" mode)
    static NN_BLEND: OnceLock<f32> = OnceLock::new();
    let nn_blend = *NN_BLEND.get_or_init(|| {
        std::env::var("NN_BLEND").ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5)
    });

    let vn = unsafe { &*(vn_ptr as *const crate::valuenet::ValueNet) };

    let board = state.board;
    let pot = state.pot + state.bets[0] + state.bets[1];
    let stacks = [state.stacks[0] - state.bets[0], state.stacks[1] - state.bets[1]];

    // Determine next street
    let n_board_cards = (0..52u8).filter(|&c| board.contains(c)).count();
    let next_street = match n_board_cards {
        3 => Street::Turn,
        4 => Street::River,
        _ => { return; }
    };

    // Enumerate ALL available next cards (not canonical)
    let available_cards: Vec<u8> = (0..52u8).filter(|&c| !board.contains(c)).collect();
    let n_cards = available_cards.len();
    if n_cards == 0 { return; }

    // --- Compute equity per card (needed for "anchor", "blend", "delta" modes) ---
    let need_equity = calibrate_mode != "off";
    let mut equity_per_card: Vec<[f32; NUM_COMBOS]> = Vec::new();
    if need_equity {
        equity_per_card.reserve(n_cards);
        for &card in &available_cards {
            let new_board = board.add(card);
            let bsd = BoardShowdown::compute(new_board, combo_map);
            let mut card_eq = [0.0f32; NUM_COMBOS];
            showdown_utility_with_precomputed(config, &bsd, state, reach, traverser, &mut card_eq);
            equity_per_card.push(card_eq);
        }
    }

    // --- Build NN batch inputs with suit isomorphism ---
    let mut batch_inputs = Vec::with_capacity(n_cards);
    let mut combo_perms: Vec<[u16; NUM_COMBOS]> = Vec::with_capacity(n_cards);

    for &card in &available_cards {
        let new_board = board.add(card);
        let (canon_board, suit_perm) = crate::iso::canonical_board(new_board);
        let combo_perm = crate::iso::combo_permutation(&suit_perm);

        let mut ranges = [Range::empty(), Range::empty()];
        for i in 0..live_count {
            let orig = combo_map.live_to_combo[i] as usize;
            let canon = combo_perm[orig] as usize;
            ranges[0].weights[canon] = reach[0][i];
            ranges[1].weights[canon] = reach[1][i];
        }
        let canon_card = crate::card::rank(card) * 4
            + suit_perm[crate::card::suit(card) as usize];
        for &ci in &CARD_COMBOS[canon_card as usize] {
            ranges[0].weights[ci as usize] = 0.0;
            ranges[1].weights[ci as usize] = 0.0;
        }

        batch_inputs.push((canon_board, ranges, pot, stacks, next_street));
        combo_perms.push(combo_perm);
    }

    // Single batched NN forward pass
    let batch_evs = vn.predict_batch(&batch_inputs, traverser);

    // --- Aggregate per-card values with calibration ---
    let inv = 1.0 / n_cards as f32;
    let trav = traverser as usize;

    for (card_idx, &card) in available_cards.iter().enumerate() {
        let nn_ev = &batch_evs[card_idx];
        let combo_perm = &combo_perms[card_idx];

        if calibrate_mode == "delta" {
            // Delta learning: NN predicts correction over equity.
            // result[c] = equity[c] + nn_delta[c]
            // If NN predicts 0 everywhere, falls back to pure equity.
            let eq_ev = &equity_per_card[card_idx];

            for i in 0..live_count {
                let orig = combo_map.live_to_combo[i] as usize;
                let (c1, c2) = combo_from_index(orig as u16);
                if c1 == card || c2 == card { continue; }
                let nn_delta = nn_ev[combo_perm[orig] as usize];
                out[i] += eq_ev[i] + nn_delta;
            }
        } else if calibrate_mode == "anchor" {
            // Equity-anchored calibration: shift NN mean to match equity mean.
            // Preserves NN's per-combo relative ordering while fixing the
            // positive-mean bias that causes pot-scaling → all-in degeneracy.
            // result[c] = nn[c] + (eq_mean - nn_mean)
            let eq_ev = &equity_per_card[card_idx];

            // Compute reach-weighted means for both NN and equity
            let mut nn_rw_sum = 0.0f32;
            let mut eq_rw_sum = 0.0f32;
            let mut reach_sum = 0.0f32;
            for i in 0..live_count {
                let orig = combo_map.live_to_combo[i] as usize;
                let (c1, c2) = combo_from_index(orig as u16);
                if c1 == card || c2 == card { continue; }
                let r = reach[trav][i];
                if r > 0.0 {
                    nn_rw_sum += r * nn_ev[combo_perm[orig] as usize];
                    eq_rw_sum += r * eq_ev[i];
                    reach_sum += r;
                }
            }
            let shift = if reach_sum > 0.0 {
                (eq_rw_sum - nn_rw_sum) / reach_sum
            } else {
                0.0
            };

            // Accumulate shifted NN values
            for i in 0..live_count {
                let orig = combo_map.live_to_combo[i] as usize;
                let (c1, c2) = combo_from_index(orig as u16);
                if c1 == card || c2 == card { continue; }
                out[i] += nn_ev[combo_perm[orig] as usize] + shift;
            }
        } else {
            // Pure NN or blend mode: accumulate raw NN values
            for i in 0..live_count {
                let orig = combo_map.live_to_combo[i] as usize;
                let (c1, c2) = combo_from_index(orig as u16);
                if c1 == card || c2 == card { continue; }
                out[i] += nn_ev[combo_perm[orig] as usize];
            }
        }
    }
    for i in 0..live_count {
        out[i] *= inv;
    }

    // Legacy blend mode: interpolate between NN and equity
    if calibrate_mode == "blend" {
        let mut equity_agg = [0.0f32; NUM_COMBOS];
        for eq_ev in &equity_per_card {
            for i in 0..live_count {
                equity_agg[i] += eq_ev[i];
            }
        }
        for i in 0..live_count {
            equity_agg[i] *= inv;
            out[i] = nn_blend * out[i] + (1.0 - nn_blend) * equity_agg[i];
        }
    }
}

// ---------------------------------------------------------------------------
// CFR+ traversal (free function for disjoint borrows)
// ---------------------------------------------------------------------------

fn cfr_traverse(
    tree: &[TreeNode],
    cfr: &[CfrData],
    arena_ptr: *mut ArenaInt,
    config: &SubgameConfig,
    sd_cache: &ShowdownCache,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &mut [[f32; NUM_COMBOS]; 2],
    pure_t_reach: &[f32; NUM_COMBOS],
    traverser: Player,
    node_id: u32,
    ancestor_perms: &[[u16; NUM_COMBOS]],
    out: &mut [f32; NUM_COMBOS],
) {
    let live_count = combo_map.live_count;

    match &tree[node_id as usize] {
        TreeNode::TerminalFold(folder) => {
            fold_utility(config, state, reach, traverser, *folder, combo_map, out);
        }
        TreeNode::TerminalShowdown => {
            showdown_utility_cached(config, sd_cache, state, reach, traverser, combo_map, out);
        }
        TreeNode::DepthLimitedLeaf { vn_ptr } => {
            #[cfg(feature = "nn")]
            {
                depth_limited_utility(*vn_ptr, config, combo_map, state, reach, traverser, out);
            }
            #[cfg(not(feature = "nn"))]
            {
                let _ = vn_ptr;
                for v in out[..live_count].iter_mut() { *v = 0.0; }
            }
        }
        TreeNode::Chance { children, iso_perms, n_actual } => {
            chance_utility(tree, cfr, arena_ptr, config, sd_cache, combo_map, state, reach, pure_t_reach, traverser,
                children, iso_perms, *n_actual, ancestor_perms, out);
        }
        TreeNode::Decision { player, actions, children, cfr_idx } => {
            let player = *player;
            let cfr_idx = *cfr_idx as usize;
            let n_actions = actions.len();
            let p = player as usize;
            let offset = cfr[cfr_idx].regret_offset;
            let regret_len = cfr[cfr_idx].regret_len();

            // Phase 1.1: Compute strategy using SoA layout: strat[action][combo].
            // At traverser nodes, decode regrets to f32 first and compute strategy
            // from the decoded buffer. This buffer is held across recursion and reused
            // for regret update, eliminating a second decode pass.
            let mut strat = pop_strat_buf();
            let mut early_regret_buf: Option<Vec<f32>> = None;
            {
                let regrets = unsafe { std::slice::from_raw_parts(arena_ptr.add(offset), regret_len) };
                let scale = cfr[cfr_idx].regret_scale.get();
                let expl_eps = config.exploration_eps;
                // Decaying regret floor: prevents early CFR+ clamping from killing actions.
                let rm_floor = if config.rm_floor > 0.0 {
                    config.rm_floor / (1.0 + 0.01 * config.current_iteration as f32)
                } else {
                    0.0
                };
                if config.softmax_temp > 0.0 {
                    // QRE softmax: σ(a) = exp(R_cum(a)/τ) / Σ exp(R_cum(a)/τ)
                    if player == traverser {
                        let mut buf = pop_regret_buf(regret_len);
                        decode_regrets_i16(regrets, scale, &mut buf);
                        softmax_match_f32(&buf, n_actions, live_count, &mut strat, config.softmax_temp);
                        early_regret_buf = Some(buf);
                    } else {
                        softmax_match_all(regrets, scale, n_actions, live_count, &mut strat, config.softmax_temp);
                    }
                } else if player == traverser {
                    let mut buf = pop_regret_buf(regret_len);
                    decode_regrets_i16(regrets, scale, &mut buf);
                    regret_match_f32(&buf, n_actions, live_count, &mut strat, expl_eps, rm_floor);
                    early_regret_buf = Some(buf);
                } else {
                    regret_match_all(regrets, scale, n_actions, live_count, &mut strat, expl_eps, rm_floor);
                }
            }

            // Frozen root: override strategy with fixed probabilities.
            // Only applies at root (cfr_idx=0) for the root player (OOP).
            let is_frozen_root = cfr_idx == 0 && config.frozen_root.is_some();
            if is_frozen_root {
                let frozen = config.frozen_root.as_ref().unwrap();
                for c in 0..live_count {
                    let orig = combo_map.live_to_combo[c] as usize;
                    let bet_frac = frozen[orig].clamp(0.0, 1.0);
                    // Action 0 = Check, Action 1 = Bet (standard root layout)
                    if n_actions >= 2 {
                        strat[0][c] = 1.0 - bet_frac;
                        strat[1][c] = bet_frac;
                        for a in 2..n_actions {
                            strat[a][c] = 0.0;
                        }
                    }
                }
            }

            // Diluted best response: mix opponent strategy with uniform.
            if player != traverser && config.opp_dilute > 0.0 {
                let delta = config.opp_dilute;
                let one_m = 1.0 - delta;
                let uni = delta / n_actions as f32;
                for a in 0..n_actions {
                    for c in 0..live_count {
                        strat[a][c] = one_m * strat[a][c] + uni;
                    }
                }
            }

            // Phase 1.3: Pooled action utilities buffer.
            let mut action_utils = pop_utils_buf();

            // RBP: Regret-Based Pruning at traverser nodes.
            // After warm-up, skip subtrees of actions with 0 strategy across ALL combos.
            // Periodic full traversals (every PRUNING_CHECK iterations) let pruned actions recover.
            const PRUNING_WARMUP: u32 = 50;
            const PRUNING_CHECK: u32 = 50;
            let mut pruned = [false; MAX_ACTIONS];
            let pruning_active = player == traverser
                && config.pruning
                && config.cfr_plus
                && config.current_iteration >= PRUNING_WARMUP
                && config.current_iteration % PRUNING_CHECK != 0;
            if pruning_active && n_actions >= 2 {
                for a in 0..n_actions {
                    let s = &strat[a];
                    let mut total = 0.0f32;
                    for c in 0..live_count {
                        total += s[c];
                    }
                    pruned[a] = total == 0.0;
                }
                // Never prune ALL actions — keep at least 1
                let n_pruned = pruned[..n_actions].iter().filter(|&&p| p).count();
                if n_pruned == n_actions {
                    pruned.fill(false);
                }
                // Zero action_utils for pruned actions (prevent NaN from pool buffers in node_util)
                for a in 0..n_actions {
                    if pruned[a] {
                        action_utils[a][..live_count].fill(0.0);
                    }
                }
            }

            // Phase 1.2: In-place reach modification with save/restore.
            // Key invariant: reach[traverser] is NEVER modified at traverser's nodes.
            // Only reach[opp] is modified at opponent nodes.
            if player != traverser {
                // Opponent node: save opp reach once, set per action, restore once.
                // Only save/restore live_count elements to avoid copying dead combos.
                // SAFETY: Only [..live_count] is used by all hot-path code.
                // SAFETY: Only [..live_count] elements are read/written.
                // Using zeroed init avoids UB from MaybeUninit::assume_init on floats.
                let mut saved_reach = [0.0f32; NUM_COMBOS];
                saved_reach[..live_count].copy_from_slice(&reach[p][..live_count]);
                for a in 0..n_actions {
                    let next_state = state.apply(actions[a]);
                    vec_mul(&mut reach[p][..live_count], &saved_reach[..live_count], &strat[a][..live_count]);
                    cfr_traverse(
                        tree, cfr, arena_ptr, config, sd_cache, combo_map, &next_state,
                        reach, pure_t_reach, traverser, children[a],
                        ancestor_perms, &mut action_utils[a],
                    );
                }
                reach[p][..live_count].copy_from_slice(&saved_reach[..live_count]);
            } else {
                // Traverser node: reach unchanged, update pure_t_reach per action.
                for a in 0..n_actions {
                    // RBP: skip subtree for fully-pruned actions (strat=0 for all combos).
                    // action_utils[a] has garbage — OK since strat[a]*garbage=0 in node_util.
                    if pruned[a] { continue; }
                    let next_state = state.apply(actions[a]);
                    if config.skip_cum_strategy {
                        // pure_t_reach is only used for cum_strategy accumulation.
                        // When skip_cum is true, skip the expensive copy+multiply.
                        cfr_traverse(
                            tree, cfr, arena_ptr, config, sd_cache, combo_map, &next_state,
                            reach, pure_t_reach, traverser, children[a],
                            ancestor_perms, &mut action_utils[a],
                        );
                    } else {
                        let mut next_pure_t = *pure_t_reach;
                        let strat_a = &strat[a];
                        for c in 0..live_count {
                            next_pure_t[c] *= strat_a[c];
                        }
                        cfr_traverse(
                            tree, cfr, arena_ptr, config, sd_cache, combo_map, &next_state,
                            reach, &next_pure_t, traverser, children[a],
                            ancestor_perms, &mut action_utils[a],
                        );
                    }
                }
            }

            // Phase 2.1: Node utility — compute directly into `out` to avoid return copy.
            if player == traverser {
                vec_mul(&mut out[..live_count], &strat[0][..live_count], &action_utils[0][..live_count]);
                for a in 1..n_actions {
                    vec_fma(&mut out[..live_count], &strat[a][..live_count], &action_utils[a][..live_count]);
                }
            } else {
                out[..live_count].copy_from_slice(&action_utils[0][..live_count]);
                for a in 1..n_actions {
                    vec_add(&mut out[..live_count], &action_utils[a][..live_count]);
                }
            }
            // iso_perms maps dead combos to index live_count — must be 0 for aggregation.
            out[live_count] = 0.0;
            let node_util = &*out;

            // Update regrets and cum_strategy for traverser.
            if player == traverser {
                // SAFETY: arena_ptr is a raw pointer to the contiguous arena.
                // Each cfr node has disjoint regret regions. cum_strategy is per-node Vec.
                // We access cum_strategy via raw pointer to avoid &mut borrow conflict with arena_ptr.
                let cfr_node = unsafe { &mut *(cfr.as_ptr().add(cfr_idx) as *mut CfrData) };

                // Phase 2.2: Cum_strategy FIRST (uses pre-update regrets via strat).
                if let Some(ref mut cum) = cfr_node.cum_strategy {
                    // t-weighting: multiply contribution by iteration number (CFR+ linear averaging)
                    let t_mult = if config.t_weight { config.current_iteration as f32 } else { 1.0 };
                    // Cum_strategy bias: boost action 0 (check/fold) contribution at ROOT ONLY.
                    // All-node bias destroys NE (4.23% expl). Root-only preserves subtree integrity.
                    let pref_cum_delta = if cfr_idx == 0 { config.pref_passive_delta } else { 1.0 };
                    // Canonical contribution
                    for c in 0..live_count {
                        let r = pure_t_reach[c] * t_mult;
                        if r <= 0.0 { continue; }
                        let base = c * n_actions;
                        for a in 0..n_actions {
                            let d = if a == 0 && pref_cum_delta > 1.0 { pref_cum_delta } else { 1.0 };
                            cum[base + a] += r * strat[a][c] * d;
                        }
                    }
                    // Isomorphic contribution via permuted indices
                    for perm in ancestor_perms {
                        for c in 0..live_count {
                            let r = pure_t_reach[c] * t_mult;
                            if r <= 0.0 { continue; }
                            let pc = perm[c] as usize;
                            if pc >= live_count { continue; } // dead-mapped combo
                            let base = pc * n_actions;
                            for a in 0..n_actions {
                                let d = if a == 0 && pref_cum_delta > 1.0 { pref_cum_delta } else { 1.0 };
                                cum[base + a] += r * strat[a][c] * d;
                            }
                        }
                    }
                }

                // Frozen root: skip regret update (strategy is fixed externally).
                // early_regret_buf will be dropped, which is fine.
                if !is_frozen_root {

                // Regret update: reuse early_regret_buf (already decoded in Phase 1.1).
                // This eliminates a second i16→f32 decode pass over regret_len elements.
                let mut regret_buf = early_regret_buf.take().unwrap();

                // Check reward: add pot-relative epsilon to check action's utility at root.
                // Pushes indifferent hands toward check → passive NE (closer to GTO+/Pio).
                // eps = pref_beta * pot. Optimal: pref_beta=8.0 (9.1% avg diff vs GTO+).
                if cfr_idx == 0 && config.pref_beta > 0.0 && n_actions >= 2 {
                    let eps = config.pref_beta * config.pot as f32;
                    for c in 0..live_count {
                        action_utils[0][c] += eps;
                    }
                }

                // Per-combo check bias at root: hand-specific NE selection.
                // Each combo gets its own bias (positive=check, negative=bet).
                if cfr_idx == 0 && n_actions >= 2 {
                    if let Some(ref bias) = config.combo_check_bias {
                        for c in 0..live_count {
                            let orig = combo_map.live_to_combo[c] as usize;
                            action_utils[0][c] += bias[orig];
                        }
                    }
                }

                // Regret update uses SoA layout: regret_buf[a * live_count + c].
                // Branchless: t_reach branch removed. Safe because zero-reach combos
                // have action_utils=0 and node_util=0, so update is a no-op.
                if ancestor_perms.is_empty() {
                    // Fast path: no isomorphic permutations — vectorized helpers
                    // Skip CFR+ clamping when softmax is active (needs negative regrets)
                    if config.cfr_plus && config.softmax_temp <= 0.0 {
                        for a in 0..n_actions {
                            if pruned[a] { continue; } // RBP: skip pruned actions
                            let r_base = a * live_count;
                            vec_regret_update_cfrplus(
                                &mut regret_buf[r_base..r_base + live_count],
                                &action_utils[a][..live_count],
                                &node_util[..live_count],
                            );
                        }
                    } else {
                        for a in 0..n_actions {
                            if pruned[a] { continue; } // RBP: skip pruned actions
                            let r_base = a * live_count;
                            vec_regret_update(
                                &mut regret_buf[r_base..r_base + live_count],
                                &action_utils[a][..live_count],
                                &node_util[..live_count],
                            );
                        }
                    }
                } else {
                    // Slow path: isomorphic permutations — vectorized canonical + scalar isomorphic.
                    // Canonical update: vectorized, updates ALL combos unconditionally.
                    // Safe: inactive combos have action_utils=0, node_util=0, so update is no-op.
                    for a in 0..n_actions {
                        if pruned[a] { continue; } // RBP: skip pruned actions
                        let r_base = a * live_count;
                        vec_regret_update(
                            &mut regret_buf[r_base..r_base + live_count],
                            &action_utils[a][..live_count],
                            &node_util[..live_count],
                        );
                    }

                    // Isomorphic update: scatter via perms (can't vectorize due to random writes).
                    let t_reach = &reach[traverser as usize];
                    for perm in ancestor_perms {
                        for c in 0..live_count {
                            if t_reach[c] <= 0.0 { continue; }
                            let pc = perm[c] as usize;
                            if pc >= live_count { continue; }
                            for a in 0..n_actions {
                                if pruned[a] { continue; } // RBP
                                regret_buf[a * live_count + pc] += action_utils[a][c] - node_util[c];
                            }
                        }
                    }

                    // CFR+ clamping — vectorized.
                    // Skip when softmax is active: softmax needs negative regrets
                    // to assign low (but nonzero) probabilities.
                    if config.cfr_plus && config.softmax_temp <= 0.0 {
                        vec_clamp_positive(&mut regret_buf[..regret_len]);
                    }
                }

                // MaxEnt regularization: add entropy bonus to regrets.
                // Bonus = τ * (-log(max(σ(a), ε))) for each action, encouraging mixing.
                // With entropy_anneal: entropy active for first 20% of iterations, then 0.
                // With entropy_root_only: apply only at root decision node (cfr_idx==0).
                if config.entropy_bonus > 0.0 && (!config.entropy_root_only || cfr_idx == 0) {
                    let base_tau = config.entropy_bonus;
                    let tau = if config.entropy_anneal && config.iterations > 0 {
                        let frac = config.current_iteration as f32 / config.iterations as f32;
                        base_tau * (1.0 - 5.0 * frac).max(0.0)
                    } else {
                        base_tau
                    };
                    for a in 0..n_actions {
                        let r_base = a * live_count;
                        for c in 0..live_count {
                            let sigma = strat[a][c].max(1e-7);
                            regret_buf[r_base + c] += tau * (-sigma.ln());
                        }
                    }
                    if config.cfr_plus && config.softmax_temp <= 0.0 {
                        vec_clamp_positive(&mut regret_buf[..regret_len]);
                    }
                }

                // Encode f32→i16 with new scale, write back to arena.
                let i16_out = unsafe { std::slice::from_raw_parts_mut(arena_ptr.add(offset), regret_len) };
                cfr[cfr_idx].regret_scale.set(encode_regrets_i16(&regret_buf, i16_out));
                push_regret_buf(regret_buf);

                } // end if !is_frozen_root (regret update skip)
            }

            // Return buffers to thread-local pool for reuse.
            push_strat_buf(strat);
            push_utils_buf(action_utils);
            // node_util is already in `out`
        }
    }
}

// ---------------------------------------------------------------------------
// Chance node — deal next card(s) during CFR
// ---------------------------------------------------------------------------

fn chance_utility(
    tree: &[TreeNode],
    cfr: &[CfrData],
    arena_ptr: *mut ArenaInt,
    config: &SubgameConfig,
    sd_cache: &ShowdownCache,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    pure_t_reach: &[f32; NUM_COMBOS],
    traverser: Player,
    children: &[(u8, u32)],
    iso_perms: &[Vec<[u16; NUM_COMBOS]>],
    n_actual: u8,
    ancestor_perms: &[[u16; NUM_COMBOS]],
    out: &mut [f32; NUM_COMBOS],
) {
    if children.is_empty() {
        for v in out.iter_mut() { *v = 0.0; }
        return;
    }

    let live_count = combo_map.live_count;

    // SAFETY: Each child subtree has disjoint CFR indices (built depth-first).
    // arena_ptr is a raw pointer — each child accesses disjoint regions via regret_offset.
    let arena_ptr_usize = arena_ptr as usize; // for Send across par_iter

    // Traverse only canonical children, composing ancestor_perms for nested chance
    let card_utils: Vec<[f32; NUM_COMBOS]> = children
        .par_iter()
        .enumerate()
        .map(|(i, &(card, child_id))| {
            let mut next_reach = *reach;
            // CARD_COMBOS uses original indices → convert to compact via combo_to_live
            for &ci in &CARD_COMBOS[card as usize] {
                let compact = combo_map.combo_to_live[ci as usize];
                if compact < 0 { continue; }
                let j = compact as usize;
                next_reach[0][j] = 0.0;
                next_reach[1][j] = 0.0;
            }
            let next_state = state.deal_street(&[card]);
            let child_ancestor = compose_perms(ancestor_perms, &iso_perms[i]);
            let ap = arena_ptr_usize as *mut ArenaInt;
            let mut result = [0.0f32; NUM_COMBOS];
            if config.skip_cum_strategy {
                cfr_traverse(tree, cfr, ap, config, sd_cache, combo_map, &next_state,
                    &mut next_reach, pure_t_reach, traverser, child_id, &child_ancestor, &mut result);
            } else {
                let mut next_pure_t = *pure_t_reach;
                for &ci in &CARD_COMBOS[card as usize] {
                    let compact = combo_map.combo_to_live[ci as usize];
                    if compact < 0 { continue; }
                    next_pure_t[compact as usize] = 0.0;
                }
                cfr_traverse(tree, cfr, ap, config, sd_cache, combo_map, &next_state,
                    &mut next_reach, &next_pure_t, traverser, child_id, &child_ancestor, &mut result);
            }
            result
        })
        .collect();

    // Aggregate utilities: canonical (identity) + isomorphic (permuted)
    for v in out.iter_mut() { *v = 0.0; }
    for (i, cu) in card_utils.iter().enumerate() {
        // Canonical contribution
        for c in 0..live_count { out[c] += cu[c]; }
        // Isomorphic contributions: use inverse direction.
        // perm = σ (forward: canonical → isomorphic).
        // For combo t in parent: utility from isomorphic subtree = cu[σ⁻¹(t)].
        // Equivalently: out[σ(c)] += cu[c] (scatter-style, avoids computing inverse).
        for perm in &iso_perms[i] {
            for c in 0..live_count {
                let pc = perm[c] as usize;
                if pc < live_count {
                    out[pc] += cu[c];
                }
            }
        }
    }
    let inv = 1.0 / n_actual as f32;
    for c in 0..live_count { out[c] *= inv; }
    // iso_perms maps dead combos to live_count — zero it for parent aggregation.
    out[live_count] = 0.0;
}

// ---------------------------------------------------------------------------
// Terminal utilities (pure functions)
// ---------------------------------------------------------------------------

pub(crate) fn fold_utility(
    config: &SubgameConfig,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    traverser: Player,
    folder: Player,
    combo_map: &ComboMap,
    out: &mut [f32; NUM_COMBOS],
) {
    let opp = 1 - traverser as usize;
    let live_count = combo_map.live_count;
    let invested = (config.stacks[traverser as usize] - state.stacks[traverser as usize]) as f32;
    let total_pot = (state.pot + state.bets[0] + state.bets[1]) as f32;
    let rake = if config.rake_pct > 0.0 {
        (total_pot * config.rake_pct).min(config.rake_cap)
    } else { 0.0 };
    let raked_pot = total_pot - rake;
    // Zero all elements including [live_count] sentinel for iso_perms dead-combo mapping.
    for v in out.iter_mut() { *v = 0.0; }

    let payoff = if folder == traverser { -invested } else { raked_pot - invested };

    // SAFETY: All indices are provably in-bounds:
    // - oc/c in 0..live_count, live_count ≤ NUM_COMBOS, reach is [NUM_COMBOS]
    // - live_to_combo has exactly live_count elements
    // - o1/o2/c1/c2 are card indices 0..51, per_card_reach is [52]
    // - combo_from_index returns cards from static table (always valid)
    unsafe {
        let opp_reach = reach.get_unchecked(opp);
        let t_reach = reach.get_unchecked(traverser as usize);
        let ltc = combo_map.live_to_combo.as_ptr();

        let mut total_opp_reach = 0.0f32;
        let mut per_card_reach = [0.0f32; 52];
        for oc in 0..live_count {
            let r = *opp_reach.get_unchecked(oc);
            if r <= 0.0 { continue; }
            let orig = *ltc.add(oc) as usize;
            let (o1, o2) = combo_from_index(orig as u16);
            total_opp_reach += r;
            *per_card_reach.get_unchecked_mut(o1 as usize) += r;
            *per_card_reach.get_unchecked_mut(o2 as usize) += r;
        }

        for c in 0..live_count {
            if *t_reach.get_unchecked(c) <= 0.0 { continue; }
            let orig = *ltc.add(c) as usize;
            let (c1, c2) = combo_from_index(orig as u16);
            let adj_opp_reach = total_opp_reach
                - *per_card_reach.get_unchecked(c1 as usize)
                - *per_card_reach.get_unchecked(c2 as usize)
                + *opp_reach.get_unchecked(c);
            *out.get_unchecked_mut(c) = payoff * adj_opp_reach;
        }
    }
}

/// Showdown utility using precomputed BoardShowdown cache.
pub(crate) fn showdown_utility_cached(
    config: &SubgameConfig,
    sd_cache: &ShowdownCache,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    traverser: Player,
    combo_map: &ComboMap,
    out: &mut [f32; NUM_COMBOS],
) {
    let board = state.board;
    if let Some(bsd) = sd_cache.get(board) {
        showdown_utility_with_precomputed(config, bsd, state, reach, traverser, out);
        return;
    }
    // Fallback: compute on the fly (shouldn't happen if cache is pre-populated)
    let bsd = BoardShowdown::compute(board, combo_map);
    showdown_utility_with_precomputed(config, &bsd, state, reach, traverser, out);
}

/// Showdown utility using O(N) two-pointer merge over strength-sorted combos.
///
/// Previous algorithm: 52×N_opp 2D prefix sum array (~104KB working set) + binary search.
/// New algorithm: three [f32; 52] stack arrays (624 bytes) + single linear scan.
///
/// Both are mathematically equivalent — same inclusion-exclusion for card removal:
///   adj_beat  = total_beat  - per_card_beat[c1]  - per_card_beat[c2]
///   adj_tie   = total_tie   - per_card_tie[c1]   - per_card_tie[c2]   + r_overlap
///   adj_total = total_opp   - per_card_total[c1]  - per_card_total[c2]  + r_overlap
///
/// Key insight: bsd.sorted is strength-ordered, so we process groups in ascending
/// strength. Beat accumulators grow as we advance through groups. Tie arrays are
/// populated per-group and cleaned via dirty tracking.
#[inline]
pub(crate) fn showdown_utility_with_precomputed(
    config: &SubgameConfig,
    bsd: &BoardShowdown,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    traverser: Player,
    out: &mut [f32; NUM_COMBOS],
) {
    let opp = 1 - traverser as usize;
    let t = traverser as usize;
    let invested = (config.stacks[traverser as usize] - state.stacks[traverser as usize]) as f32;
    let total_pot = (state.pot + state.bets[0] + state.bets[1]) as f32;
    let rake = if config.rake_pct > 0.0 {
        (total_pot * config.rake_pct).min(config.rake_cap)
    } else { 0.0 };
    let raked_pot = total_pot - rake;
    let half_pot = raked_pot * 0.5;
    for v in out.iter_mut() { *v = 0.0; }

    // SAFETY: All indices are provably in-bounds:
    // - ci is compact index < live_count ≤ NUM_COMBOS, reach is [NUM_COMBOS]
    // - c1/c2 are card indices 0..51, per_card arrays are [52]
    // - sorted[j] with j in group_start..i, i ≤ sorted.len()
    // - n_dirty ≤ 52 (at most 52 distinct cards)
    unsafe {
    let opp_reach = reach.get_unchecked(opp);
    let t_reach = reach.get_unchecked(t);

    // Pass 1: total opp reach per card (same as fold_utility's inclusion-exclusion)
    let mut per_card_total = [0.0f32; 52];
    let mut total_opp_reach = 0.0f32;
    for &(ci, _, c1, c2) in &bsd.sorted {
        let r = *opp_reach.get_unchecked(ci as usize);
        if r <= 0.0 { continue; }
        total_opp_reach += r;
        *per_card_total.get_unchecked_mut(c1 as usize) += r;
        *per_card_total.get_unchecked_mut(c2 as usize) += r;
    }

    // Pass 2: strength groups, ascending
    let mut per_card_beat = [0.0f32; 52];
    let mut total_beat = 0.0f32;

    // Tie tracking (per group, cleaned via dirty list)
    let mut per_card_tie = [0.0f32; 52];
    let mut tie_dirty = [false; 52];
    let mut tie_dirty_list = [0u8; 52];
    let mut n_dirty: usize;

    let sorted = &bsd.sorted;
    let n = sorted.len();
    let mut i = 0;

    while i < n {
        let group_strength = sorted.get_unchecked(i).1;
        let group_start = i;

        // Find end of this strength group
        while i < n && sorted.get_unchecked(i).1 == group_strength {
            i += 1;
        }

        // Compute tie opp reach for this group
        let mut total_tie = 0.0f32;
        n_dirty = 0;

        for j in group_start..i {
            let (ci, _, c1, c2) = *sorted.get_unchecked(j);
            let r = *opp_reach.get_unchecked(ci as usize);
            if r <= 0.0 { continue; }
            total_tie += r;
            let c1u = c1 as usize;
            let c2u = c2 as usize;
            if !*tie_dirty.get_unchecked(c1u) {
                *tie_dirty.get_unchecked_mut(c1u) = true;
                *tie_dirty_list.get_unchecked_mut(n_dirty) = c1;
                n_dirty += 1;
            }
            *per_card_tie.get_unchecked_mut(c1u) += r;
            if !*tie_dirty.get_unchecked(c2u) {
                *tie_dirty.get_unchecked_mut(c2u) = true;
                *tie_dirty_list.get_unchecked_mut(n_dirty) = c2;
                n_dirty += 1;
            }
            *per_card_tie.get_unchecked_mut(c2u) += r;
        }

        // Process my combos in this group
        for j in group_start..i {
            let (ci, _, c1, c2) = *sorted.get_unchecked(j);
            if *t_reach.get_unchecked(ci as usize) <= 0.0 { continue; }

            let c1u = c1 as usize;
            let c2u = c2 as usize;
            let r_overlap = *opp_reach.get_unchecked(ci as usize);

            let adj_beat = total_beat - *per_card_beat.get_unchecked(c1u) - *per_card_beat.get_unchecked(c2u);
            let adj_tie = total_tie - *per_card_tie.get_unchecked(c1u) - *per_card_tie.get_unchecked(c2u) + r_overlap;
            let adj_total = total_opp_reach - *per_card_total.get_unchecked(c1u) - *per_card_total.get_unchecked(c2u) + r_overlap;

            *out.get_unchecked_mut(ci as usize) =
                raked_pot * adj_beat + half_pot * adj_tie - invested * adj_total;
        }

        // Transfer tie data → beat accumulators + clean up (fused).
        // Tie per_card values ARE the beat deltas (same opp reach, same cards).
        total_beat += total_tie;
        for k in 0..n_dirty {
            let c = *tie_dirty_list.get_unchecked(k) as usize;
            *per_card_beat.get_unchecked_mut(c) += *per_card_tie.get_unchecked(c);
            *per_card_tie.get_unchecked_mut(c) = 0.0;
            *tie_dirty.get_unchecked_mut(c) = false;
        }
    }
    } // unsafe
}

// ---------------------------------------------------------------------------
// Read-only traversals (eval_avg_strategy, br_traverse)
// ---------------------------------------------------------------------------

pub(crate) fn eval_avg_strategy(
    tree: &[TreeNode],
    cfr: &[CfrData],
    arena: &[ArenaInt],
    config: &SubgameConfig,
    sd_cache: &ShowdownCache,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    player: Player,
    node_id: u32,
) -> [f32; NUM_COMBOS] {
    let live_count = combo_map.live_count;
    match &tree[node_id as usize] {
        TreeNode::TerminalFold(folder) => {
            let mut util = [0.0f32; NUM_COMBOS];
            fold_utility(config, state, reach, player, *folder, combo_map, &mut util);
            util
        }
        TreeNode::TerminalShowdown => {
            let mut util = [0.0f32; NUM_COMBOS];
            showdown_utility_cached(config, sd_cache, state, reach, player, combo_map, &mut util);
            util
        }
        TreeNode::DepthLimitedLeaf { vn_ptr } => {
            #[allow(unused_mut)]
            let mut util = [0.0f32; NUM_COMBOS];
            #[cfg(feature = "nn")]
            {
                depth_limited_utility(*vn_ptr, config, combo_map, state, reach, player, &mut util);
            }
            #[cfg(not(feature = "nn"))]
            { let _ = vn_ptr; }
            util
        }
        TreeNode::Chance { children, iso_perms, n_actual } => {
            chance_eval_parallel(tree, cfr, arena, config, combo_map, state, reach, children, iso_perms, *n_actual,
                |t, c, a, cfg, cm, st, r, nid| eval_avg_strategy(t, c, a, cfg, sd_cache, cm, st, r, player, nid))
        }
        TreeNode::Decision { player: acting_player, actions, children, cfr_idx } => {
            let acting = *acting_player;
            let cfr_idx = *cfr_idx as usize;
            let n_act = actions.len();
            let avg_strat = compute_avg_strat_flat(&cfr[cfr_idx], arena, config.exploration_eps, config.softmax_temp);
            let mut node_util = [0.0f32; NUM_COMBOS];
            let p = acting as usize;

            if acting == player {
                for a in 0..n_act {
                    let next_state = state.apply(actions[a]);
                    let action_util = eval_avg_strategy(tree, cfr, arena, config, sd_cache, combo_map, &next_state, reach, player, children[a]);
                    for c in 0..live_count {
                        if reach[0][c] <= 0.0 && reach[1][c] <= 0.0 { continue; }
                        node_util[c] += avg_strat[c * n_act + a] * action_util[c];
                    }
                }
            } else {
                for a in 0..n_act {
                    let next_state = state.apply(actions[a]);
                    let mut next_reach = *reach;
                    for c in 0..live_count {
                        if next_reach[p][c] <= 0.0 { continue; }
                        next_reach[p][c] *= avg_strat[c * n_act + a];
                    }
                    let action_util = eval_avg_strategy(tree, cfr, arena, config, sd_cache, combo_map, &next_state, &next_reach, player, children[a]);
                    for c in 0..live_count {
                        if reach[0][c] <= 0.0 && reach[1][c] <= 0.0 { continue; }
                        node_util[c] += action_util[c];
                    }
                }
            }
            node_util
        }
    }
}

pub(crate) fn br_traverse(
    tree: &[TreeNode],
    cfr: &[CfrData],
    arena: &[ArenaInt],
    config: &SubgameConfig,
    sd_cache: &ShowdownCache,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    br_player: Player,
    node_id: u32,
) -> [f32; NUM_COMBOS] {
    let live_count = combo_map.live_count;
    match &tree[node_id as usize] {
        TreeNode::TerminalFold(folder) => {
            let mut util = [0.0f32; NUM_COMBOS];
            fold_utility(config, state, reach, br_player, *folder, combo_map, &mut util);
            util
        }
        TreeNode::TerminalShowdown => {
            let mut util = [0.0f32; NUM_COMBOS];
            showdown_utility_cached(config, sd_cache, state, reach, br_player, combo_map, &mut util);
            util
        }
        TreeNode::DepthLimitedLeaf { vn_ptr } => {
            #[allow(unused_mut)]
            let mut util = [0.0f32; NUM_COMBOS];
            #[cfg(feature = "nn")]
            {
                depth_limited_utility(*vn_ptr, config, combo_map, state, reach, br_player, &mut util);
            }
            #[cfg(not(feature = "nn"))]
            { let _ = vn_ptr; }
            util
        }
        TreeNode::Chance { children, iso_perms, n_actual } => {
            chance_eval_parallel(tree, cfr, arena, config, combo_map, state, reach, children, iso_perms, *n_actual,
                |t, c, a, cfg, cm, st, r, nid| br_traverse(t, c, a, cfg, sd_cache, cm, st, r, br_player, nid))
        }
        TreeNode::Decision { player: acting_player, actions, children, cfr_idx } => {
            let acting = *acting_player;
            let cfr_idx = *cfr_idx as usize;
            let n_act = actions.len();
            let avg_strat = compute_avg_strat_flat(&cfr[cfr_idx], arena, config.exploration_eps, config.softmax_temp);

            if acting == br_player {
                let mut node_util = [f32::NEG_INFINITY; NUM_COMBOS];
                for a in 0..n_act {
                    let next_state = state.apply(actions[a]);
                    let action_util = br_traverse(tree, cfr, arena, config, sd_cache, combo_map, &next_state, reach, br_player, children[a]);
                    for c in 0..live_count {
                        if reach[br_player as usize][c] > 0.0 {
                            node_util[c] = node_util[c].max(action_util[c]);
                        }
                    }
                }
                for c in 0..live_count {
                    if node_util[c] == f32::NEG_INFINITY { node_util[c] = 0.0; }
                }
                node_util
            } else {
                let mut node_util = [0.0f32; NUM_COMBOS];
                let p = acting as usize;
                for a in 0..n_act {
                    let next_state = state.apply(actions[a]);
                    let mut next_reach = *reach;
                    for c in 0..live_count {
                        if next_reach[p][c] <= 0.0 { continue; }
                        next_reach[p][c] *= avg_strat[c * n_act + a];
                    }
                    let action_util = br_traverse(tree, cfr, arena, config, sd_cache, combo_map, &next_state, &next_reach, br_player, children[a]);
                    for c in 0..live_count {
                        if reach[0][c] <= 0.0 && reach[1][c] <= 0.0 { continue; }
                        node_util[c] += action_util[c];
                    }
                }
                node_util
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel chance evaluation for read-only traversals
// ---------------------------------------------------------------------------

pub(crate) fn chance_eval_parallel<F>(
    tree: &[TreeNode],
    cfr: &[CfrData],
    arena: &[ArenaInt],
    config: &SubgameConfig,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    children: &[(u8, u32)],
    iso_perms: &[Vec<[u16; NUM_COMBOS]>],
    n_actual: u8,
    traverse_fn: F,
) -> [f32; NUM_COMBOS]
where
    F: Fn(&[TreeNode], &[CfrData], &[ArenaInt], &SubgameConfig, &ComboMap, &GameState, &[[f32; NUM_COMBOS]; 2], u32) -> [f32; NUM_COMBOS]
        + Sync,
{
    if children.is_empty() { return [0.0f32; NUM_COMBOS]; }

    let live_count = combo_map.live_count;

    let card_utils: Vec<[f32; NUM_COMBOS]> = children
        .par_iter()
        .map(|&(card, child_id)| {
            let mut next_reach = *reach;
            for &ci in &CARD_COMBOS[card as usize] {
                let compact = combo_map.combo_to_live[ci as usize];
                if compact < 0 { continue; }
                let i = compact as usize;
                next_reach[0][i] = 0.0;
                next_reach[1][i] = 0.0;
            }
            let next_state = state.deal_street(&[card]);
            traverse_fn(tree, cfr, arena, config, combo_map, &next_state, &next_reach, child_id)
        })
        .collect();

    // Aggregate: canonical + isomorphic (inverse direction, same as chance_utility)
    let mut util = [0.0f32; NUM_COMBOS];
    for (i, cu) in card_utils.iter().enumerate() {
        for c in 0..live_count { util[c] += cu[c]; }
        for perm in &iso_perms[i] {
            for c in 0..live_count {
                let pc = perm[c] as usize;
                if pc < live_count {
                    util[pc] += cu[c];
                }
            }
        }
    }
    let inv = 1.0 / n_actual as f32;
    for c in 0..live_count { util[c] *= inv; }
    util
}

// ---------------------------------------------------------------------------
// Helper: compute flat average strategy
// ---------------------------------------------------------------------------

pub(crate) fn compute_avg_strat_flat(cfr_node: &CfrData, arena: &[ArenaInt], expl_eps: f32, softmax_temp: f32) -> Vec<f32> {
    let n_act = cfr_node.n_actions;
    let live_count = cfr_node.live_count;
    let one_m = 1.0 - expl_eps;
    let uni = expl_eps / n_act as f32;
    let force_current = FORCE_CURRENT_STRATEGY.with(|f| f.get());
    if !force_current && cfr_node.cum_strategy.is_some() {
        let cum = cfr_node.cum_strategy.as_ref().unwrap();
        let mut avg_strat = vec![0.0f32; live_count * n_act];
        let uniform = 1.0 / n_act as f32;
        for c in 0..live_count {
            let base = c * n_act;
            let mut total = 0.0f32;
            for a in 0..n_act {
                total += cum[base + a];
            }
            if total > 1e-7 {
                let inv = 1.0 / total;
                for a in 0..n_act {
                    avg_strat[base + a] = one_m * cum[base + a] * inv + uni;
                }
            } else {
                for a in 0..n_act {
                    avg_strat[base + a] = uniform;
                }
            }
        }
        avg_strat
    } else if softmax_temp > 0.0 {
        // QRE mode: softmax of cumulative regrets
        let regrets = &arena[cfr_node.regret_offset..cfr_node.regret_offset + cfr_node.regret_len()];
        let scale = cfr_node.regret_scale.get();
        let inv_temp = 1.0 / softmax_temp;
        let mut avg_strat = vec![0.0f32; live_count * n_act];
        for c in 0..live_count {
            let out_base = c * n_act;
            let mut max_r = f32::NEG_INFINITY;
            for a in 0..n_act {
                let r = regrets[a * live_count + c] as f32 * scale;
                if r > max_r { max_r = r; }
            }
            let mut sum_exp = 0.0f32;
            for a in 0..n_act {
                let r = regrets[a * live_count + c] as f32 * scale;
                let e = ((r - max_r) * inv_temp).exp();
                avg_strat[out_base + a] = e;
                sum_exp += e;
            }
            let inv = 1.0 / sum_exp;
            for a in 0..n_act {
                avg_strat[out_base + a] *= inv;
            }
        }
        avg_strat
    } else {
        // Fallback: regret-matched strategy from current i16 regrets (SoA layout).
        let regrets = &arena[cfr_node.regret_offset..cfr_node.regret_offset + cfr_node.regret_len()];
        let scale = cfr_node.regret_scale.get();
        let mut avg_strat = vec![0.0f32; live_count * n_act];
        let uniform = 1.0 / n_act as f32;
        for c in 0..live_count {
            let out_base = c * n_act;
            let mut pos_sum = 0.0f32;
            for a in 0..n_act {
                let r = (regrets[a * live_count + c] as f32 * scale).max(0.0);
                avg_strat[out_base + a] = r;
                pos_sum += r;
            }
            if pos_sum > 0.0 {
                let inv = 1.0 / pos_sum;
                for a in 0..n_act {
                    avg_strat[out_base + a] = one_m * avg_strat[out_base + a] * inv + uni;
                }
            } else {
                for a in 0..n_act {
                    avg_strat[out_base + a] = uniform;
                }
            }
        }
        avg_strat
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card;

    fn river_board() -> Hand {
        Hand::new()
            .add(card(12, 3)) // As
            .add(card(11, 2)) // Kh
            .add(card(8, 1))  // Td
            .add(card(3, 0))  // 5c
            .add(card(0, 3))  // 2s
    }

    #[test]
    fn test_solver_creates() {
        let config = SubgameConfig {
            board: river_board(),
            pot: 20,
            stacks: [90, 90],
            ranges: [Range::uniform(), Range::uniform()],
            iterations: 10,
            street: Street::River,
            warmup_frac: 0.5,
            bet_config: None,
            dcfr: true,
            cfr_plus: true, skip_cum_strategy: false, dcfr_mode: DcfrMode::Standard,
            depth_limit: None, rake_pct: 0.0, rake_cap: 0.0, exploration_eps: 0.0, entropy_bonus: 0.0, entropy_anneal: false, entropy_root_only: false, opp_dilute: 0.0, softmax_temp: 0.0, current_iteration: 0, use_iso: true, rm_floor: 0.0, alternating: false, t_weight: false, frozen_root: None, check_bias: 0.0, pref_passive_delta: 1.0, pref_beta: 0.0, pref_beta_all_nodes: false, pruning: false, combo_check_bias: None, frozen_warmup: 0, unfreeze_decay: 1.0,
        };
        let mut solver = SubgameSolver::new(config);
        solver.solve();
        assert!(solver.num_decision_nodes() > 0);
    }

    #[test]
    fn test_river_solve_small_range() {
        let oop_range = Range::parse("AA,KK").unwrap();
        let ip_range = Range::parse("QQ,JJ").unwrap();

        let config = SubgameConfig {
            board: river_board(),
            pot: 20,
            stacks: [90, 90],
            ranges: [oop_range, ip_range],
            iterations: 100,
            street: Street::River,
            warmup_frac: 0.5,
            bet_config: None,
            dcfr: true,
            cfr_plus: true, skip_cum_strategy: false, dcfr_mode: DcfrMode::Standard,
            depth_limit: None, rake_pct: 0.0, rake_cap: 0.0, exploration_eps: 0.0, entropy_bonus: 0.0, entropy_anneal: false, entropy_root_only: false, opp_dilute: 0.0, softmax_temp: 0.0, current_iteration: 0, use_iso: true, rm_floor: 0.0, alternating: false, t_weight: false, frozen_root: None, check_bias: 0.0, pref_passive_delta: 1.0, pref_beta: 0.0, pref_beta_all_nodes: false, pruning: false, combo_check_bias: None, frozen_warmup: 0, unfreeze_decay: 1.0,
        };
        let mut solver = SubgameSolver::new(config);
        solver.solve();
        assert!(solver.get_strategy(&[]).is_some(), "root node should exist");
    }

    #[test]
    fn test_exploitability_basic() {
        let board = river_board();
        let config = SubgameConfig {
            board,
            pot: 100,
            stacks: [100, 100],
            ranges: [Range::parse("AA,76o").unwrap(), Range::parse("TT").unwrap()],
            iterations: 200,
            street: Street::River,
            warmup_frac: 0.5,
            bet_config: None,
            dcfr: true,
            cfr_plus: true, skip_cum_strategy: false, dcfr_mode: DcfrMode::Standard,
            depth_limit: None, rake_pct: 0.0, rake_cap: 0.0, exploration_eps: 0.0, entropy_bonus: 0.0, entropy_anneal: false, entropy_root_only: false, opp_dilute: 0.0, softmax_temp: 0.0, current_iteration: 0, use_iso: true, rm_floor: 0.0, alternating: false, t_weight: false, frozen_root: None, check_bias: 0.0, pref_passive_delta: 1.0, pref_beta: 0.0, pref_beta_all_nodes: false, pruning: false, combo_check_bias: None, frozen_warmup: 0, unfreeze_decay: 1.0,
        };
        let mut solver = SubgameSolver::new(config);
        solver.solve();

        let expl = solver.exploitability();
        let expl_pct = solver.exploitability_pct();
        println!("Exploitability: {:.2} chips ({:.2}% pot)", expl, expl_pct);
        assert!(expl >= -1.0, "exploitability should be >= ~0, got {}", expl);
    }

    #[test]
    fn test_river_nuts_vs_air() {
        let oop_range = Range::parse("AA").unwrap();
        let ip_range = Range::parse("33").unwrap();

        let config = SubgameConfig {
            board: river_board(),
            pot: 20,
            stacks: [90, 90],
            ranges: [oop_range, ip_range],
            iterations: 200,
            street: Street::River,
            warmup_frac: 0.5,
            bet_config: None,
            dcfr: true,
            cfr_plus: true, skip_cum_strategy: false, dcfr_mode: DcfrMode::Standard,
            depth_limit: None, rake_pct: 0.0, rake_cap: 0.0, exploration_eps: 0.0, entropy_bonus: 0.0, entropy_anneal: false, entropy_root_only: false, opp_dilute: 0.0, softmax_temp: 0.0, current_iteration: 0, use_iso: true, rm_floor: 0.0, alternating: false, t_weight: false, frozen_root: None, check_bias: 0.0, pref_passive_delta: 1.0, pref_beta: 0.0, pref_beta_all_nodes: false, pruning: false, combo_check_bias: None, frozen_warmup: 0, unfreeze_decay: 1.0,
        };
        let mut solver = SubgameSolver::new(config);
        solver.solve();
        assert!(solver.num_decision_nodes() > 0);
    }
}
