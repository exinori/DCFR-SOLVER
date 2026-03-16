/// EGT solver — Excessive Gap Technique with Dilated Entropy DGF.
///
/// Provably converges to the unique Maximum Entropy Nash Equilibrium.
///
/// Key difference from OMD (previous attempt):
/// - Separate gradient computation (full tree traversal for counterfactual values)
/// - Coupled proximal operator (bottom-up partition → top-down strategy)
/// - Convex combination updates: x_{t+1} = (1-τ)x_t + τ·x̂
/// - Dual smoothing parameters μ₁, μ₂ that decrease each iteration

use crate::card::{NUM_COMBOS, CARD_COMBOS};
use crate::cfr::{
    SubgameSolver, TreeNode, CfrData, ComboMap, ShowdownCache, SubgameConfig,
    fold_utility, showdown_utility_cached,
};
use crate::game::{GameState, Player, OOP, IP};
use rayon::prelude::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Buffer pool — reuse [f32; NUM_COMBOS] arrays to avoid heap churn
// Same pattern as STRAT_POOL/UTILS_POOL in cfr.rs.
// ---------------------------------------------------------------------------

thread_local! {
    static EGT_BUF_POOL: RefCell<Vec<Box<[f32; NUM_COMBOS]>>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn pop_buf() -> Box<[f32; NUM_COMBOS]> {
    EGT_BUF_POOL.with(|pool| {
        pool.borrow_mut().pop().unwrap_or_else(|| Box::new([0.0f32; NUM_COMBOS]))
    })
}

#[inline]
fn push_buf(buf: Box<[f32; NUM_COMBOS]>) {
    EGT_BUF_POOL.with(|pool| pool.borrow_mut().push(buf));
}

// ---------------------------------------------------------------------------
// Per-node EGT strategy
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct EgtNodeStrat {
    /// Behavioral strategy: [action * live_count + combo] (action-major).
    strat: Vec<f32>,
    n_actions: usize,
    live_count: usize,
}

impl EgtNodeStrat {
    fn new_uniform(n_actions: usize, live_count: usize) -> Self {
        let uniform = 1.0 / n_actions as f32;
        let size = n_actions * live_count;
        EgtNodeStrat {
            strat: vec![uniform; size],
            n_actions,
            live_count,
        }
    }

    #[inline(always)]
    fn get(&self, action: usize, combo: usize) -> f32 {
        unsafe { *self.strat.get_unchecked(action * self.live_count + combo) }
    }

    #[inline(always)]
    fn set(&mut self, action: usize, combo: usize, val: f32) {
        unsafe { *self.strat.get_unchecked_mut(action * self.live_count + combo) = val; }
    }
}

// ---------------------------------------------------------------------------
// EGT state: per-player strategy storage
// ---------------------------------------------------------------------------

pub struct EgtState {
    strats: [Vec<Option<EgtNodeStrat>>; 2],
}

impl EgtState {
    fn new(tree: &[TreeNode], cfr: &[CfrData]) -> Self {
        let n_cfr = cfr.len();
        let mut strats = [
            (0..n_cfr).map(|_| None).collect::<Vec<_>>(),
            (0..n_cfr).map(|_| None).collect::<Vec<_>>(),
        ];
        for node in tree {
            if let TreeNode::Decision { player, actions, cfr_idx, .. } = node {
                let p = *player as usize;
                let idx = *cfr_idx as usize;
                if strats[p][idx].is_none() {
                    strats[p][idx] = Some(EgtNodeStrat::new_uniform(
                        actions.len(),
                        cfr[idx].live_count,
                    ));
                }
            }
        }
        EgtState { strats }
    }
}

// ---------------------------------------------------------------------------
// Infoset tree structure — parent-child relationships for coupled prox
// ---------------------------------------------------------------------------

struct InfosetTree {
    /// Bottom-up topological order (leaves first).
    topo_order: Vec<u32>,
    /// children[cfr_idx][action] = child cfr_idx (-1 if no same-player child).
    children: Vec<Vec<i32>>,
}

impl InfosetTree {
    fn build(tree: &[TreeNode], cfr: &[CfrData], player: Player) -> Self {
        let n_cfr = cfr.len();
        let mut children = vec![Vec::new(); n_cfr];
        let mut is_player_node = vec![false; n_cfr];

        for node in tree {
            if let TreeNode::Decision { player: p, actions, cfr_idx, .. } = node {
                if *p == player {
                    let idx = *cfr_idx as usize;
                    is_player_node[idx] = true;
                    if children[idx].is_empty() {
                        children[idx] = vec![-1i32; actions.len()];
                    }
                }
            }
        }

        Self::walk_tree(tree, player, 0, -1, 0, &mut children);

        // Topological sort via BFS from roots, then reverse
        let mut in_degree = vec![0u32; n_cfr];
        for idx in 0..n_cfr {
            if !is_player_node[idx] { continue; }
            for &child in &children[idx] {
                if child >= 0 { in_degree[child as usize] += 1; }
            }
        }

        let mut top_down = Vec::new();
        for idx in 0..n_cfr {
            if is_player_node[idx] && in_degree[idx] == 0 {
                top_down.push(idx as u32);
            }
        }
        let mut i = 0;
        while i < top_down.len() {
            let idx = top_down[i] as usize;
            for &child in &children[idx] {
                if child >= 0 {
                    in_degree[child as usize] -= 1;
                    if in_degree[child as usize] == 0 {
                        top_down.push(child as u32);
                    }
                }
            }
            i += 1;
        }

        let topo_order: Vec<u32> = top_down.into_iter().rev().collect();
        InfosetTree { topo_order, children }
    }

    fn walk_tree(
        tree: &[TreeNode], player: Player, node_id: u32,
        ancestor_cfr: i32, ancestor_action: usize,
        children: &mut [Vec<i32>],
    ) {
        match &tree[node_id as usize] {
            TreeNode::TerminalFold(_) | TreeNode::TerminalShowdown
            | TreeNode::DepthLimitedLeaf { .. } => {}
            TreeNode::Chance { children: ch, .. } => {
                for &(_, child_id) in ch {
                    Self::walk_tree(tree, player, child_id, ancestor_cfr, ancestor_action, children);
                }
            }
            TreeNode::Decision { player: p, actions, children: ch, cfr_idx } => {
                let idx = *cfr_idx as usize;
                if *p == player {
                    if ancestor_cfr >= 0 && children[ancestor_cfr as usize].len() > ancestor_action {
                        children[ancestor_cfr as usize][ancestor_action] = idx as i32;
                    }
                    for a in 0..actions.len() {
                        Self::walk_tree(tree, player, ch[a], idx as i32, a, children);
                    }
                } else {
                    for a in 0..actions.len() {
                        Self::walk_tree(tree, player, ch[a], ancestor_cfr, ancestor_action, children);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient buffer
// ---------------------------------------------------------------------------

struct GradientBuffer {
    /// gradients[cfr_idx] = [n_actions * live_count] action-major.
    data: Vec<Option<Vec<f32>>>,
}

impl GradientBuffer {
    fn new(n_cfr: usize) -> Self {
        GradientBuffer { data: (0..n_cfr).map(|_| None).collect() }
    }

    fn ensure(&mut self, cfr_idx: usize, n_actions: usize, live_count: usize) {
        if self.data[cfr_idx].is_none() {
            self.data[cfr_idx] = Some(vec![0.0f32; n_actions * live_count]);
        }
    }

    fn get_mut(&mut self, cfr_idx: usize) -> &mut [f32] {
        self.data[cfr_idx].as_mut().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Gradient traversal — computes counterfactual values (read-only)
// Uses output parameter to avoid 5.3KB return-by-value per recursive call.
// Uses buffer pool to avoid heap allocation per node.
// ---------------------------------------------------------------------------

/// Compute counterfactual values for `target_player`.
/// Stores per-action gradients in `grad_buf` for target_player's decision nodes.
/// Writes the node utility vector to `out`.
fn gradient_traverse(
    tree: &[TreeNode],
    cfr: &[CfrData],
    config: &SubgameConfig,
    sd_cache: &ShowdownCache,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    target_player: Player,
    node_id: u32,
    my_strats: &[Option<EgtNodeStrat>],
    opp_strats: &[Option<EgtNodeStrat>],
    grad_buf_ptr: *mut GradientBuffer,
    live_count: usize,
    out: &mut [f32; NUM_COMBOS],
) {
    match &tree[node_id as usize] {
        TreeNode::TerminalFold(folder) => {
            fold_utility(config, state, reach, target_player, *folder, combo_map, out);
        }
        TreeNode::TerminalShowdown => {
            showdown_utility_cached(config, sd_cache, state, reach, target_player, combo_map, out);
        }
        TreeNode::DepthLimitedLeaf { .. } => {
            for v in out[..live_count].iter_mut() { *v = 0.0; }
        }
        TreeNode::Chance { children, iso_perms, n_actual } => {
            if children.is_empty() {
                for v in out[..live_count].iter_mut() { *v = 0.0; }
                return;
            }

            // SAFETY: Each child subtree has disjoint CFR indices (built depth-first).
            // grad_buf_ptr points to stable memory — each child writes to disjoint slots.
            let gbp_usize = grad_buf_ptr as usize;

            let card_utils: Vec<[f32; NUM_COMBOS]> = children
                .par_iter()
                .enumerate()
                .map(|(_i, &(card, child_id))| {
                    let mut next_reach = *reach;
                    for &ci in &CARD_COMBOS[card as usize] {
                        let compact = combo_map.combo_to_live[ci as usize];
                        if compact < 0 { continue; }
                        let idx = compact as usize;
                        next_reach[0][idx] = 0.0;
                        next_reach[1][idx] = 0.0;
                    }
                    let next_state = state.deal_street(&[card]);
                    let gbp = gbp_usize as *mut GradientBuffer;
                    let mut result = [0.0f32; NUM_COMBOS];
                    gradient_traverse(
                        tree, cfr, config, sd_cache, combo_map,
                        &next_state, &next_reach, target_player, child_id,
                        my_strats, opp_strats, gbp, live_count,
                        &mut result,
                    );
                    result
                })
                .collect();

            // Aggregate: canonical + isomorphic contributions
            for v in out[..live_count].iter_mut() { *v = 0.0; }
            for (i, cu) in card_utils.iter().enumerate() {
                for c in 0..live_count { out[c] += cu[c]; }
                for perm in &iso_perms[i] {
                    for c in 0..live_count { out[c] += cu[perm[c] as usize]; }
                }
            }
            let inv = 1.0 / *n_actual as f32;
            for c in 0..live_count { out[c] *= inv; }
            // iso_perms maps dead combos to live_count — zero it for parent aggregation.
            out[live_count] = 0.0;
        }
        TreeNode::Decision { player: acting_player, actions, children, cfr_idx } => {
            let acting = *acting_player;
            let cfr_idx_val = *cfr_idx as usize;
            let n_act = actions.len();
            let grad_buf = unsafe { &mut *grad_buf_ptr };

            if acting == target_player {
                grad_buf.ensure(cfr_idx_val, n_act, live_count);

                // Use buffer pool for per-action utilities
                let mut action_bufs: Vec<Box<[f32; NUM_COMBOS]>> = Vec::with_capacity(n_act);
                for a in 0..n_act {
                    let mut buf = pop_buf();
                    let next_state = state.apply(actions[a]);
                    gradient_traverse(
                        tree, cfr, config, sd_cache, combo_map,
                        &next_state, reach, target_player, children[a],
                        my_strats, opp_strats, grad_buf_ptr, live_count,
                        &mut buf,
                    );
                    action_bufs.push(buf);
                }

                let strat = my_strats[cfr_idx_val].as_ref().unwrap();
                let grad = grad_buf.get_mut(cfr_idx_val);
                for v in out[..live_count].iter_mut() { *v = 0.0; }

                for c in 0..live_count {
                    for a in 0..n_act {
                        grad[a * live_count + c] = action_bufs[a][c];
                        out[c] += strat.get(a, c) * action_bufs[a][c];
                    }
                }

                // Return buffers to pool
                for buf in action_bufs { push_buf(buf); }
            } else {
                let opp_node = opp_strats[cfr_idx_val].as_ref().unwrap();
                for v in out[..live_count].iter_mut() { *v = 0.0; }
                let p = acting as usize;
                let mut au = pop_buf();
                for a in 0..n_act {
                    let next_state = state.apply(actions[a]);
                    let mut next_reach = *reach;
                    for c in 0..live_count {
                        if next_reach[p][c] <= 0.0 { continue; }
                        next_reach[p][c] *= opp_node.get(a, c);
                    }
                    gradient_traverse(
                        tree, cfr, config, sd_cache, combo_map,
                        &next_state, &next_reach, target_player, children[a],
                        my_strats, opp_strats, grad_buf_ptr, live_count,
                        &mut au,
                    );
                    for c in 0..live_count { out[c] += au[c]; }
                }
                push_buf(au);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Proximal operator for Dilated Entropy — coupled bottom-up/top-down
// ---------------------------------------------------------------------------

/// Helper: compute effective gradient ĝ for action `a` at combo `c`.
/// ĝ[a] = g_full[a] - child_current_value + V_child_prox
#[inline]
fn effective_gradient(
    a: usize, c: usize, live_count: usize,
    grad_data: &[f32],
    child_idx: i32,
    v_values: &[Vec<f32>],
    grad_buf: &GradientBuffer,
    cfr: &[CfrData],
    current_strats: &[Option<EgtNodeStrat>],
) -> f32 {
    let g = grad_data[a * live_count + c];
    if child_idx >= 0 && !v_values[child_idx as usize].is_empty() {
        let child = child_idx as usize;
        let child_grad = grad_buf.data[child].as_ref().unwrap();
        let child_lc = cfr[child].live_count;
        let child_nact = cfr[child].n_actions;
        let child_cur = current_strats[child].as_ref().unwrap();
        let mut child_val = 0.0f32;
        for b in 0..child_nact {
            child_val += child_cur.get(b, c) * child_grad[b * child_lc + c];
        }
        g - child_val + v_values[child][c]
    } else {
        g
    }
}

/// Compute smoothed best response via the Dilated Entropy proximal operator.
///
/// The prox solves: argmin_x { <g, x> + (1/η) · Ω(x) } on the treeplex.
///
/// `grad_buf` contains full-subtree gradients from gradient_traverse.
/// The prox must subtract the child's current-strategy value and add the prox-optimal
/// V_child to avoid double-counting.
///
/// `current_strats`: the player's CURRENT strategies (for computing child_current_value)
/// `eta`: step size = 1/μ
/// `out_strats`: prox output strategies
fn prox_solve(
    cfr: &[CfrData],
    info_tree: &InfosetTree,
    grad_buf: &GradientBuffer,
    eta: f32,
    current_strats: &[Option<EgtNodeStrat>],
    out_strats: *mut Option<EgtNodeStrat>,
) {
    let n_cfr = cfr.len();
    let mut v_values: Vec<Vec<f32>> = vec![Vec::new(); n_cfr];

    // Bottom-up pass: compute V_j for each infoset (leaves first)
    for &idx in &info_tree.topo_order {
        let idx = idx as usize;
        let grad_data = match &grad_buf.data[idx] {
            Some(d) => d,
            None => continue,
        };
        let live_count = cfr[idx].live_count;
        let n_act = cfr[idx].n_actions;
        let mut v = vec![0.0f32; live_count];

        for c in 0..live_count {
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..n_act {
                let ghat = effective_gradient(
                    a, c, live_count, grad_data,
                    info_tree.children[idx][a], &v_values, grad_buf, cfr, current_strats,
                );
                let val = -eta * ghat;
                if val > max_val { max_val = val; }
            }

            if max_val == f32::NEG_INFINITY { v[c] = 0.0; continue; }

            let mut sum_exp = 0.0f64;
            for a in 0..n_act {
                let ghat = effective_gradient(
                    a, c, live_count, grad_data,
                    info_tree.children[idx][a], &v_values, grad_buf, cfr, current_strats,
                );
                let val = -eta * ghat;
                sum_exp += ((val - max_val) as f64).exp();
            }
            let log_z = max_val as f64 + sum_exp.ln();
            v[c] = -(1.0 / eta) as f64 as f32 * log_z as f32;
        }
        v_values[idx] = v;
    }

    // Top-down pass: compute behavioral strategies (roots first = reverse topo)
    for &idx in info_tree.topo_order.iter().rev() {
        let idx = idx as usize;
        let grad_data = match &grad_buf.data[idx] {
            Some(d) => d,
            None => continue,
        };
        let live_count = cfr[idx].live_count;
        let n_act = cfr[idx].n_actions;

        let strat = unsafe { &mut *out_strats.add(idx) };
        let strat = strat.as_mut().unwrap();

        for c in 0..live_count {
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..n_act {
                let ghat = effective_gradient(
                    a, c, live_count, grad_data,
                    info_tree.children[idx][a], &v_values, grad_buf, cfr, current_strats,
                );
                let val = -eta * ghat;
                if val > max_val { max_val = val; }
            }

            let mut sum = 0.0f64;
            for a in 0..n_act {
                let ghat = effective_gradient(
                    a, c, live_count, grad_data,
                    info_tree.children[idx][a], &v_values, grad_buf, cfr, current_strats,
                );
                let val = -eta * ghat;
                let w = ((val - max_val) as f64).exp();
                strat.set(a, c, w as f32);
                sum += w;
            }

            if sum > 0.0 {
                let inv = 1.0 / sum as f32;
                for a in 0..n_act {
                    strat.set(a, c, (strat.get(a, c) * inv).max(1e-10));
                }
            } else {
                let uniform = 1.0 / n_act as f32;
                for a in 0..n_act { strat.set(a, c, uniform); }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convex combination in SEQUENCE-FORM space: x_{t+1} = (1-τ)·x_t + τ·x̂
// Must be done in sequence-form (not behavioral) for EGT convergence.
// At non-root infosets, the parent reach differs between x_t and x̂,
// so behavioral convex combination ≠ sequence-form convex combination.
// ---------------------------------------------------------------------------

fn convex_combine_seq(
    strats: &mut [Option<EgtNodeStrat>],
    prox_result: &[Option<EgtNodeStrat>],
    info_tree: &InfosetTree,
    cfr: &[CfrData],
    tau: f32,
) {
    let one_m_tau = 1.0 - tau;
    let n_cfr = cfr.len();

    // Build parent map: parent[idx] = (parent_cfr_idx, action) or (-1, 0) for roots
    let mut parent: Vec<(i32, usize)> = vec![(-1, 0); n_cfr];
    for idx in 0..n_cfr {
        for (a, &child) in info_tree.children[idx].iter().enumerate() {
            if child >= 0 {
                parent[child as usize] = (idx as i32, a);
            }
        }
    }

    // Pass 1: compute per-combo reach probabilities (top-down, original strategies)
    let mut reach_cur: Vec<Vec<f32>> = vec![Vec::new(); n_cfr];
    let mut reach_prox: Vec<Vec<f32>> = vec![Vec::new(); n_cfr];

    for &idx in info_tree.topo_order.iter().rev() {
        let idx = idx as usize;
        let strat = match &strats[idx] {
            Some(s) => s,
            None => continue,
        };
        let lc = strat.live_count;
        let (par_idx, par_act) = parent[idx];

        if par_idx < 0 {
            reach_cur[idx] = vec![1.0; lc];
            reach_prox[idx] = vec![1.0; lc];
        } else {
            let pi = par_idx as usize;
            let par_s = strats[pi].as_ref().unwrap();
            let par_p = prox_result[pi].as_ref().unwrap();
            let mut rc = Vec::with_capacity(lc);
            let mut rp = Vec::with_capacity(lc);
            for c in 0..lc {
                rc.push(reach_cur[pi][c] * par_s.get(par_act, c));
                rp.push(reach_prox[pi][c] * par_p.get(par_act, c));
            }
            reach_cur[idx] = rc;
            reach_prox[idx] = rp;
        }
    }

    // Pass 2: update behavioral strategies via sequence-form convex combination
    for idx in 0..n_cfr {
        let (strat, prox) = match (&mut strats[idx], &prox_result[idx]) {
            (Some(s), Some(p)) => (s, p),
            _ => continue,
        };
        if reach_cur[idx].is_empty() { continue; }
        let lc = strat.live_count;
        let n_act = strat.n_actions;

        for c in 0..lc {
            let rc = reach_cur[idx][c];
            let rp = reach_prox[idx][c];
            let new_reach = one_m_tau * rc + tau * rp;

            if new_reach < 1e-30 {
                let uniform = 1.0 / n_act as f32;
                for a in 0..n_act { strat.set(a, c, uniform); }
                continue;
            }

            let inv_reach = 1.0 / new_reach;
            for a in 0..n_act {
                let seq_new = one_m_tau * rc * strat.get(a, c) + tau * rp * prox.get(a, c);
                strat.set(a, c, (seq_new * inv_reach).max(1e-10));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EGT main loop
// ---------------------------------------------------------------------------

impl SubgameSolver {
    pub fn egt_solve<F: FnMut(u32, &SubgameSolver)>(&mut self, mut callback: F) {
        let reach = self.initial_reach();
        let root_state = self.root_state();

        // Build tree if not already built
        if self.tree.is_empty() {
            let mut action_seq = Vec::with_capacity(20);
            self.root_id = self.build_tree(&root_state, &mut action_seq);
            self.remap_iso_perms();
            self.precompute_showdowns();
        }

        let live_count = self.combo_map.live_count;
        let n_cfr = self.cfr.len();

        let mut egt = EgtState::new(&self.tree, &self.cfr);

        let info_tree_oop = InfosetTree::build(&self.tree, &self.cfr, OOP);
        let info_tree_ip = InfosetTree::build(&self.tree, &self.cfr, IP);

        let total = self.config.iterations;
        let mut next_report = 1u32;

        // Operator norm estimate: max utility swing = stack (worst case: fold vs win all-in)
        let l_estimate = self.config.stacks[0] as f32;

        // Dilated entropy diameter: ln(|V|) where |V| = number of terminal nodes.
        // This is the standard upper bound from Kroer et al. 2020.
        // (NOT Σ ln(n_actions) over all infosets — that's unweighted and scales with tree size.)
        let n_terminal = self.tree.iter().filter(|n| {
            matches!(n, TreeNode::TerminalFold(_) | TreeNode::TerminalShowdown | TreeNode::DepthLimitedLeaf { .. })
        }).count().max(2);
        let omega_max = (n_terminal as f32).ln();
        // mu_floor: prevents eta=1/mu from causing numerical overflow.
        // eta = 1/mu, so mu_floor = 1/eta_max. With f32, eta up to ~1e6 is safe
        // (exp(-1e6 * x) just saturates to 0 in softmax).
        let mu_floor = 1e-6_f32;

        // μ initialization: L * Ω (theoretical value from Kroer 2018)
        let mut mu1 = l_estimate * omega_max;
        let mut mu2 = mu1;

        let mut n_dec = [0usize; 2];
        for node in &self.tree {
            if let TreeNode::Decision { player, .. } = node {
                n_dec[*player as usize] += 1;
            }
        }

        eprintln!("  EGT: L_est={:.1}, Ω=ln({})={:.2}, μ₁={:.1}, μ₂={:.1}, nodes={}, oop_dec={}, ip_dec={}",
            l_estimate, n_terminal, omega_max, mu1, mu2, self.tree.len(), n_dec[0], n_dec[1]);

        let mut arena_allocated = false;

        let mut prox_oop = EgtState::new(&self.tree, &self.cfr);
        let mut prox_ip = EgtState::new(&self.tree, &self.cfr);

        let mut grad_buf_oop = GradientBuffer::new(n_cfr);
        let mut grad_buf_ip = GradientBuffer::new(n_cfr);

        // Scratch buffer for root utility (gradient_traverse output, unused)
        let mut root_util = pop_buf();

        // Best-iterate tracking: save strategy with lowest exploitability
        let mut best_expl = f32::MAX;
        let mut best_cum: Vec<Option<Vec<f32>>> = Vec::new();


        // Clear cum_strategy — EGT uses copy_final_strategy (last iterate)
        for d in self.cfr.iter_mut() {
            d.cum_strategy = None;
        }

        for iter in 0..total {
            self.iteration = iter + 1;

            let tau = 2.0 / (iter as f32 + 3.0);

                // Step 1: Smoothed BR for IP
                {
                    let oop_strats = &egt.strats[OOP as usize];
                    let ip_strats = &egt.strats[IP as usize];
                    gradient_traverse(
                        &self.tree, &self.cfr, &self.config, &self.sd_cache,
                        &self.combo_map, &root_state, &reach, IP, self.root_id,
                        ip_strats, oop_strats, &mut grad_buf_ip as *mut GradientBuffer, live_count,
                        &mut root_util,
                    );
                }
                for idx in 0..n_cfr {
                    if let Some(ref mut g) = grad_buf_ip.data[idx] {
                        for v in g.iter_mut() { *v = -*v; }
                    }
                }
                let eta_ip = 1.0 / mu1;
                {
                    let ip_cur = &egt.strats[IP as usize];
                    let prox_ip_ptr = prox_ip.strats[IP as usize].as_mut_ptr();
                    prox_solve(&self.cfr, &info_tree_ip, &grad_buf_ip, eta_ip, ip_cur, prox_ip_ptr);
                }

                // Step 2: Smoothed BR for OOP (against IP prox result)
                {
                    let prox_ip_strats = &prox_ip.strats[IP as usize];
                    let oop_strats = &egt.strats[OOP as usize];
                    gradient_traverse(
                        &self.tree, &self.cfr, &self.config, &self.sd_cache,
                        &self.combo_map, &root_state, &reach, OOP, self.root_id,
                        oop_strats, prox_ip_strats, &mut grad_buf_oop as *mut GradientBuffer, live_count,
                        &mut root_util,
                    );
                }
                for idx in 0..n_cfr {
                    if let Some(ref mut g) = grad_buf_oop.data[idx] {
                        for v in g.iter_mut() { *v = -*v; }
                    }
                }
                let eta_oop = 1.0 / mu2;
                {
                    let oop_cur = &egt.strats[OOP as usize];
                    let prox_oop_ptr = prox_oop.strats[OOP as usize].as_mut_ptr();
                    prox_solve(&self.cfr, &info_tree_oop, &grad_buf_oop, eta_oop, oop_cur, prox_oop_ptr);
                }

                // Step 3: Convex combination (sequence-form space)
                convex_combine_seq(
                    &mut egt.strats[OOP as usize],
                    &prox_oop.strats[OOP as usize],
                    &info_tree_oop, &self.cfr, tau,
                );
                convex_combine_seq(
                    &mut egt.strats[IP as usize],
                    &prox_ip.strats[IP as usize],
                    &info_tree_ip, &self.cfr, tau,
                );

                // Step 4: Update μ — O(1/t) annealing schedule
                // mu = mu0 * k / (k + t) where mu0 = L * Omega, k = 20
                //
                // Standard EGT schedule mu *= (1-tau) gives O(1/t²) decrease,
                // which diverges in practice when mu gets small enough for
                // the prox to produce near-pure strategies.
                //
                // O(1/t) annealing is slower but stable. The strategy tracks
                // the gradually changing QRE target as mu decreases.
                // Converges to QRE(mu(T)) which approaches NE as T → ∞.
                let mu0 = l_estimate * omega_max;
                let k = 20.0_f32;
                let mu_new = mu0 * k / (k + iter as f32 + 1.0);
                mu1 = mu_new.max(mu_floor);
                mu2 = mu1;

            // Progress callback with best-iterate tracking
            if self.iteration >= next_report || self.iteration == total {
                if !arena_allocated {
                    let total_elements: usize = self.cfr.iter()
                        .map(|d| d.live_count * d.n_actions)
                        .sum();
                    self.arena = crate::cfr::RegretArena::new(total_elements);
                    let mut offset = 0usize;
                    for data in self.cfr.iter_mut() {
                        data.regret_offset = offset;
                        offset += data.live_count * data.n_actions;
                    }
                    arena_allocated = true;
                }
                self.copy_final_strategy(&egt);
                let expl = self.exploitability_pct();
                if expl < best_expl {
                    best_expl = expl;
                    best_cum = self.cfr.iter().map(|d| d.cum_strategy.clone()).collect();
                }
                callback(self.iteration, self);
                while next_report <= self.iteration {
                    next_report = next_report.saturating_mul(2);
                }
            }
        }

        // Restore best-iterate strategy
        if !best_cum.is_empty() {
            for (i, d) in self.cfr.iter_mut().enumerate() {
                if i < best_cum.len() {
                    d.cum_strategy = best_cum[i].take();
                }
            }
            eprintln!("  EGT: restored best iterate (expl={:.4}%)", best_expl);
        }

        push_buf(root_util);
    }

    /// Copy the final EGT iterate directly to cum_strategy (overwrite).
    /// EGT's convergence guarantee is for the last iterate, not the average.
    fn copy_final_strategy(&mut self, egt: &EgtState) {
        for node in &self.tree {
            if let TreeNode::Decision { player, cfr_idx, actions, .. } = node {
                let p = *player as usize;
                let idx = *cfr_idx as usize;
                let n_act = actions.len();
                let lc = self.cfr[idx].live_count;

                self.cfr[idx].ensure_cum_strategy();
                if let Some(ref mut cum) = self.cfr[idx].cum_strategy {
                    let strat = egt.strats[p][idx].as_ref().unwrap();
                    for c in 0..lc {
                        for a in 0..n_act {
                            cum[c * n_act + a] = strat.get(a, c);
                        }
                    }
                }
            }
        }
    }
}
