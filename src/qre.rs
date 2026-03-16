/// QRE Fixed-Point solver — homotopy continuation
///
/// σ(a|I,c) = exp(λ · CF(a|I,c)) / Σ_b exp(λ · CF(b|I,c))
/// λ = precision: 0 → uniform, ∞ → NE.
/// Homotopy: gradually increase λ from near-zero to target.

use crate::card::{NUM_COMBOS, CARD_COMBOS};
use crate::cfr::{
    SubgameSolver, TreeNode, CfrData, ComboMap, ShowdownCache, SubgameConfig,
    fold_utility, showdown_utility_cached,
};
use crate::game::{GameState, Player, OOP, IP};
use rayon::prelude::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Buffer pool
// ---------------------------------------------------------------------------

thread_local! {
    static QRE_BUF_POOL: RefCell<Vec<Box<[f32; NUM_COMBOS]>>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn pop_buf() -> Box<[f32; NUM_COMBOS]> {
    QRE_BUF_POOL.with(|pool| {
        pool.borrow_mut().pop().unwrap_or_else(|| Box::new([0.0f32; NUM_COMBOS]))
    })
}

#[inline]
fn push_buf(buf: Box<[f32; NUM_COMBOS]>) {
    QRE_BUF_POOL.with(|pool| pool.borrow_mut().push(buf));
}

// ---------------------------------------------------------------------------
// Per-node QRE strategy (action-major layout)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct QreStrat {
    strat: Vec<f32>,
    n_actions: usize,
    live_count: usize,
}

impl QreStrat {
    fn new_uniform(n_actions: usize, live_count: usize) -> Self {
        let uniform = 1.0 / n_actions as f32;
        QreStrat {
            strat: vec![uniform; n_actions * live_count],
            n_actions,
            live_count,
        }
    }

    #[inline(always)]
    fn get(&self, action: usize, combo: usize) -> f32 {
        unsafe { *self.strat.get_unchecked(action * self.live_count + combo) }
    }
}

struct QreState {
    strats: [Vec<Option<QreStrat>>; 2],
}

impl QreState {
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
                    strats[p][idx] = Some(QreStrat::new_uniform(
                        actions.len(),
                        cfr[idx].live_count,
                    ));
                }
            }
        }
        QreState { strats }
    }
}

// ---------------------------------------------------------------------------
// Gradient buffer — stores per-action CF values
// ---------------------------------------------------------------------------

struct GradBuf {
    data: Vec<Option<Vec<f32>>>,
}

impl GradBuf {
    fn new(n_cfr: usize) -> Self {
        GradBuf {
            data: (0..n_cfr).map(|_| None).collect(),
        }
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
// CF traversal — computes counterfactual values for target_player
// ---------------------------------------------------------------------------

fn cf_traverse(
    tree: &[TreeNode],
    cfr: &[CfrData],
    config: &SubgameConfig,
    sd_cache: &ShowdownCache,
    combo_map: &ComboMap,
    state: &GameState,
    reach: &[[f32; NUM_COMBOS]; 2],
    target_player: Player,
    node_id: u32,
    my_strats: &[Option<QreStrat>],
    opp_strats: &[Option<QreStrat>],
    grad_buf_ptr: *mut GradBuf,
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
            let gbp_usize = grad_buf_ptr as usize;
            let card_utils: Vec<[f32; NUM_COMBOS]> = children
                .par_iter()
                .enumerate()
                .map(|(_i, &(card, child_id))| {
                    let mut next_reach = *reach;
                    for &ci in &CARD_COMBOS[card as usize] {
                        let compact = combo_map.combo_to_live[ci as usize];
                        if compact < 0 { continue; }
                        next_reach[0][compact as usize] = 0.0;
                        next_reach[1][compact as usize] = 0.0;
                    }
                    let next_state = state.deal_street(&[card]);
                    let gbp = gbp_usize as *mut GradBuf;
                    let mut result = [0.0f32; NUM_COMBOS];
                    cf_traverse(
                        tree, cfr, config, sd_cache, combo_map,
                        &next_state, &next_reach, target_player, child_id,
                        my_strats, opp_strats, gbp, live_count,
                        &mut result,
                    );
                    result
                })
                .collect();

            for v in out[..live_count].iter_mut() { *v = 0.0; }
            for (i, cu) in card_utils.iter().enumerate() {
                for c in 0..live_count { out[c] += cu[c]; }
                for perm in &iso_perms[i] {
                    for c in 0..live_count { out[c] += cu[perm[c] as usize]; }
                }
            }
            let inv = 1.0 / *n_actual as f32;
            for c in 0..live_count { out[c] *= inv; }
            out[live_count] = 0.0;
        }
        TreeNode::Decision { player: acting_player, actions, children, cfr_idx } => {
            let acting = *acting_player;
            let cfr_idx_val = *cfr_idx as usize;
            let n_act = actions.len();
            let grad_buf = unsafe { &mut *grad_buf_ptr };

            if acting == target_player {
                grad_buf.ensure(cfr_idx_val, n_act, live_count);

                let mut action_bufs: Vec<Box<[f32; NUM_COMBOS]>> = Vec::with_capacity(n_act);
                for a in 0..n_act {
                    let mut buf = pop_buf();
                    let next_state = state.apply(actions[a]);
                    cf_traverse(
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
                    cf_traverse(
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
// Softmax on CF values: σ(a,c) = exp(λ·CF(a,c)) / Σ exp(λ·CF(b,c))
// ---------------------------------------------------------------------------

fn compute_softmax_target(
    strats: &[Option<QreStrat>],
    grad_buf: &GradBuf,
    lambda: f32,
) -> Vec<Option<Vec<f32>>> {
    strats.iter().enumerate().map(|(idx, strat_opt)| {
        let strat = match strat_opt {
            Some(s) => s,
            None => return None,
        };
        let grad = match &grad_buf.data[idx] {
            Some(g) => g,
            None => return None,
        };
        let n_act = strat.n_actions;
        let lc = strat.live_count;
        let mut target = vec![0.0f32; n_act * lc];

        for c in 0..lc {
            // log-sum-exp trick for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..n_act {
                let v = lambda * grad[a * lc + c];
                if v > max_val { max_val = v; }
            }

            if !max_val.is_finite() {
                let u = 1.0 / n_act as f32;
                for a in 0..n_act { target[a * lc + c] = u; }
                continue;
            }

            let mut sum_exp = 0.0f32;
            for a in 0..n_act {
                sum_exp += (lambda * grad[a * lc + c] - max_val).exp();
            }

            if sum_exp <= 0.0 || !sum_exp.is_finite() {
                let u = 1.0 / n_act as f32;
                for a in 0..n_act { target[a * lc + c] = u; }
                continue;
            }

            let inv_sum = 1.0 / sum_exp;
            for a in 0..n_act {
                target[a * lc + c] = (lambda * grad[a * lc + c] - max_val).exp() * inv_sum;
            }
        }
        Some(target)
    }).collect()
}

/// Apply damped update: strat = (1-α)·strat + α·target
fn apply_damped_update(
    strats: &mut [Option<QreStrat>],
    targets: &[Option<Vec<f32>>],
    alpha: f32,
) {
    for (idx, strat_opt) in strats.iter_mut().enumerate() {
        let strat = match strat_opt.as_mut() {
            Some(s) => s,
            None => continue,
        };
        let target = match &targets[idx] {
            Some(t) => t,
            None => continue,
        };
        for (s, t) in strat.strat.iter_mut().zip(target.iter()) {
            *s = (1.0 - alpha) * *s + alpha * *t;
        }
    }
}

// ---------------------------------------------------------------------------
// Main QRE solve
// ---------------------------------------------------------------------------

impl SubgameSolver {
    pub fn qre_solve<F: FnMut(u32, &SubgameSolver)>(
        &mut self,
        lambda_target: f32,
        damping: f32,
        anneal: bool,
        mut callback: F,
    ) {
        let reach = self.initial_reach();
        let root_state = self.root_state();

        if self.tree.is_empty() {
            let mut action_seq = Vec::with_capacity(20);
            self.root_id = self.build_tree(&root_state, &mut action_seq);
            self.remap_iso_perms();
            self.precompute_showdowns();
        }

        let live_count = self.combo_map.live_count;
        let n_cfr = self.cfr.len();

        let mut qre = QreState::new(&self.tree, &self.cfr);

        let total = self.config.iterations;
        let mut next_report = 1u32;

        let mut grad_buf_oop = GradBuf::new(n_cfr);
        let mut grad_buf_ip = GradBuf::new(n_cfr);
        let mut root_util = pop_buf();

        // Late-stage averaging: accumulate strategies from second half of iterations
        let avg_start = total / 2;
        let mut avg_strats: [Vec<Option<Vec<f32>>>; 2] = [
            (0..n_cfr).map(|_| None).collect(),
            (0..n_cfr).map(|_| None).collect(),
        ];
        for p in 0..2 {
            for (idx, s) in qre.strats[p].iter().enumerate() {
                if let Some(strat) = s {
                    avg_strats[p][idx] = Some(vec![0.0f32; strat.strat.len()]);
                }
            }
        }
        let mut avg_count = 0u32;

        let mut arena_allocated = false;
        let mut best_expl = f32::MAX;
        let mut best_cum: Vec<Option<Vec<f32>>> = Vec::new();

        for d in self.cfr.iter_mut() {
            d.cum_strategy = None;
        }

        // Homotopy: geometric growth from lambda_start to lambda_target
        let lambda_start = if anneal { (lambda_target * 0.001).max(1e-7) } else { lambda_target };
        let growth = if anneal && lambda_target > lambda_start {
            (lambda_target / lambda_start).powf(1.0 / total as f32)
        } else {
            1.0
        };
        let mut cur_lambda = lambda_start;

        eprintln!("  QRE2: target λ={}, start λ={:.7}, damping={}, growth={:.6}, anneal={}, nodes={}, live={}",
            lambda_target, lambda_start, damping, growth, anneal, self.tree.len(), live_count);

        for iter in 0..total {
            self.iteration = iter + 1;

            if anneal && iter > 0 {
                cur_lambda = (cur_lambda * growth).min(lambda_target);
            }

            // Adaptive damping: reduce alpha as lambda increases to prevent oscillation
            let alpha = if anneal {
                (damping / (1.0 + cur_lambda * 50.0).sqrt()).max(0.05)
            } else {
                damping
            };

            // Sequential (Gauss-Seidel): OOP first, then IP against updated OOP
            {
                let oop_strats = &qre.strats[OOP as usize];
                let ip_strats = &qre.strats[IP as usize];
                cf_traverse(
                    &self.tree, &self.cfr, &self.config, &self.sd_cache,
                    &self.combo_map, &root_state, &reach, OOP, self.root_id,
                    oop_strats, ip_strats, &mut grad_buf_oop as *mut GradBuf, live_count,
                    &mut root_util,
                );
            }
            let targets_oop = compute_softmax_target(&qre.strats[OOP as usize], &grad_buf_oop, cur_lambda);
            apply_damped_update(&mut qre.strats[OOP as usize], &targets_oop, alpha);

            {
                let ip_strats = &qre.strats[IP as usize];
                let oop_strats = &qre.strats[OOP as usize];
                cf_traverse(
                    &self.tree, &self.cfr, &self.config, &self.sd_cache,
                    &self.combo_map, &root_state, &reach, IP, self.root_id,
                    ip_strats, oop_strats, &mut grad_buf_ip as *mut GradBuf, live_count,
                    &mut root_util,
                );
            }
            let targets_ip = compute_softmax_target(&qre.strats[IP as usize], &grad_buf_ip, cur_lambda);
            apply_damped_update(&mut qre.strats[IP as usize], &targets_ip, alpha);

            // Accumulate for late-stage averaging
            if iter >= avg_start {
                for p in 0..2 {
                    for (idx, s) in qre.strats[p].iter().enumerate() {
                        if let Some(strat) = s {
                            if let Some(ref mut cum) = avg_strats[p][idx] {
                                for (c, v) in cum.iter_mut().zip(strat.strat.iter()) {
                                    *c += *v;
                                }
                            }
                        }
                    }
                }
                avg_count += 1;
            }

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

                // Use late-stage average if available, otherwise current iterate
                if avg_count > 0 {
                    self.qre_copy_averaged_strategy(&avg_strats);
                } else {
                    self.qre_copy_strategy(&qre);
                }
                let expl = self.exploitability_pct();
                if expl < best_expl {
                    best_expl = expl;
                    best_cum = self.cfr.iter().map(|d| d.cum_strategy.clone()).collect();
                }
                eprintln!("    λ={:.6} α={:.4}", cur_lambda, alpha);
                callback(self.iteration, self);
                while next_report <= self.iteration {
                    next_report = next_report.saturating_mul(2);
                }
            }
        }

        if !best_cum.is_empty() {
            for (i, d) in self.cfr.iter_mut().enumerate() {
                if i < best_cum.len() {
                    d.cum_strategy = best_cum[i].take();
                }
            }
            eprintln!("  QRE2: restored best iterate (expl={:.4}%)", best_expl);
        }

        push_buf(root_util);
    }

    /// Copy late-stage averaged strategy to cum_strategy
    fn qre_copy_averaged_strategy(&mut self, avg_strats: &[Vec<Option<Vec<f32>>>; 2]) {
        for node in &self.tree {
            if let TreeNode::Decision { player, cfr_idx, actions, .. } = node {
                let p = *player as usize;
                let idx = *cfr_idx as usize;
                let n_act = actions.len();
                let lc = self.cfr[idx].live_count;

                self.cfr[idx].ensure_cum_strategy();
                if let Some(ref mut cum) = self.cfr[idx].cum_strategy {
                    if let Some(ref avg) = avg_strats[p][idx] {
                        for c in 0..lc {
                            let mut sum = 0.0f32;
                            for a in 0..n_act {
                                sum += avg[a * lc + c];
                            }
                            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                            if inv > 0.0 {
                                for a in 0..n_act {
                                    cum[c * n_act + a] = avg[a * lc + c] * inv;
                                }
                            } else {
                                let u = 1.0 / n_act as f32;
                                for a in 0..n_act {
                                    cum[c * n_act + a] = u;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Copy current QRE iterate to cum_strategy for exploitability measurement
    fn qre_copy_strategy(&mut self, qre: &QreState) {
        for node in &self.tree {
            if let TreeNode::Decision { player, cfr_idx, actions, .. } = node {
                let p = *player as usize;
                let idx = *cfr_idx as usize;
                let n_act = actions.len();
                let lc = self.cfr[idx].live_count;

                self.cfr[idx].ensure_cum_strategy();
                if let Some(ref mut cum) = self.cfr[idx].cum_strategy {
                    let strat = qre.strats[p][idx].as_ref().unwrap();
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
