# DCFR Solver

A high-performance GTO (Game Theory Optimal) poker solver written in Rust, built from scratch.

No dependencies on existing solver engines. No wrappers around open-source libraries. Every line of the CFR engine, game tree, hand evaluator, and isomorphism logic was written from zero.

## The Journey

This solver started as a question: *Can we build a poker solver from nothing?*

The answer took months of research, hundreds of experiments, and more dead ends than we can count. Here's how it happened.

### Phase 1: The First Solve

We began with a textbook CFR+ implementation. The first successful river solve produced a strategy with 0.0021% exploitability — mathematically near-perfect. Turn+River followed at 0.0002% exploitability over 10K iterations. Then full Flop+Turn+River: 103K decision nodes, 0.016% exploitability.

The core algorithm worked. But it was slow.

### Phase 2: Making It Fast

The initial solver ran at 1.18 iterations per second.

What followed was a systematic optimization campaign — one change at a time, each benchmarked independently:

- **SoA memory layout**: Restructured strategy data from Array-of-Structs to Struct-of-Arrays. Enabled SIMD auto-vectorization across all hot loops. (+10.3%)
- **Range-aware compact mapping**: Instead of iterating over all 1,326 possible card combinations, we skip board-blocked and zero-weight combos. Reduced the working set from 1,176 to ~500 entries. Every hot loop runs 2x faster. (+48.2%)
- **SIMD vectorization**: Hand-tuned loop structures for NEON/SSE auto-vectorization. Verified with objdump: `fadd.4s`, `fmul.4s`, `fmaxnm.4s`. (+11.5%)
- **Unsafe terminal evaluation**: Bounds-check elimination in fold/showdown utilities — the 53% CPU hotspot. (+7.9%)
- **Return-by-ref + slow-path vectorization**: Eliminated 5.3KB memmoves per node. Vectorized the 95% of traverser nodes that use ancestor permutations. (+13.0%)

**Total: +163.8% improvement.** 1.18 → 3.12 iter/s.

Zero accuracy loss. All tests passing.

### Phase 3: The Isomorphism Bug

Suit isomorphism reduces game tree size by recognizing that permuting suits doesn't change strategy. On a board like Td9d6h, swapping clubs and spades produces an equivalent game.

It worked perfectly — on most boards. But paired boards (9h9c5d) plateaued at 0.65% exploitability, and monotone boards (Ts8s3s) at 0.73%. Normal boards converged to 0.016%.

The hunt took days. We wrote differential tests comparing ISO ON vs OFF. We tested turn-only, river-only, flop-only. We checked multiplicity invariants, canonical card mappings, permutation composition.

The root cause: `chance_utility` aggregation was applying the forward permutation instead of the inverse. For transpositions (swapping two suits), forward = inverse, so it worked. For 3-cycles in S3 (monotone boards with three equivalent suits), forward != inverse. The aggregation silently scattered utility values to wrong combos.

One-line fix. Monotone boards dropped from 0.73% to normal convergence. All boards, all textures, verified.

### Phase 4: The Multiple Equilibria Problem

With correctness established, we turned to validation. Our solver's exploitability was excellent (0.016%), but per-hand betting frequencies differed across solvers by ~12-15%.

Was this a bug? No. This is the Multiple Nash Equilibria problem. When a hand is indifferent between actions (EV difference ≈ 0), any mixing ratio is valid. Different CFR implementations converge to different valid equilibria. The strategies look different but are equally unexploitable.

We proved this definitively: 241 of 499 hands had EV gaps under 0.1 chips. These indifferent hands account for all the difference. Pure strategies (100% bet or 100% check) match 100% across all solvers tested.

### Phase 5: 30+ Experiments

We tried everything to control which equilibrium the solver converges to.

#### Algorithm Experiments

| Algorithm | Description | Result |
|-----------|-------------|--------|
| **DCFR (Discounted CFR)** | Discount past regrets/strategies, weight recent data | **Adopted. 0.016% exploitability, fastest convergence** |
| **CFR+** | Clamp negative regrets to zero | Same equilibrium as DCFR, slightly slower |
| **Vanilla CFR** | Textbook baseline | Converges but slower than DCFR |
| **HS-DCFR(30)** | Hyperparameter schedule from 2024 paper | 16% fewer iterations but offset by compute cost — net zero |
| **Binary Trick** | HS-DCFR variant requiring cum_strategy | Best convergence but 2x memory — impractical |
| **EGT (Excessive Gap Technique)** | Gradient descent + proximal updates for MaxEnt NE | Implemented. 0.22% expl. 2x slower than CFR |
| **QRE v1 (Softmax CFR)** | Softmax on cumulative regrets | 0.38% expl. Diverges after 10K iter |
| **QRE v2 (Fixed-Point Iteration)** | Softmax on instantaneous CF values, Gauss-Seidel order | Implemented. 0.22% expl. Guarantees unique equilibrium |
| **LCFR (Linear CFR)** | Linear-weighted averaging | Marginal improvement only |

#### NE Selection Experiments

Every attempt to control equilibrium selection:

| Approach | Description | Avg Diff | Verdict |
|----------|-------------|----------|---------|
| **Baseline DCFR** | No modifications | 12.3% | Baseline |
| Alpha/Gamma grid (18 configs) | Exhaustive DCFR parameter search | 12.3~39.5% | All alpha>=1.5 identical |
| i32 high-precision regrets | 4x precision to rule out quantization | 12.7% | Not the cause |
| Exploration epsilon (0.01~0.05) | Off-policy exploration | 14~19% | Worse |
| Entropy (all nodes) | Entropy bonus everywhere | 34% expl | Destructive |
| Entropy (root-only) | Entropy bonus at root only | Overall freq matched (21.5%) | Per-hand accuracy poor (17.7%) |
| Entropy annealing | Entropy early, remove gradually | 0% donk revert | DCFR discounting erases early learning |
| rm_floor (decaying) | Minimum regret threshold | 0.1~0.4% | No effect |
| Passive tiebreak blend | Blend toward check | 16.4% | Worse |
| Smooth (aggregate blend) | Strategy smoothing | 23.9% | Much worse |
| Two-phase (50/50) | Aggressive first, passive second | 16.3% | Worse |
| Two-phase inverted | Opposite order | 26.8% | Much worse |
| Check bias (0.1~100) | Additive bonus to check action | 12.5~12.8% | DCFR washes it out |
| EGT MaxEnt | Maximum entropy equilibrium | 16.6% | Worse |
| QRE softmax (5K iter) | Quantal response equilibrium | 14.4% | Worse |
| Frozen root (reference data injection) | Fix root strategy to reference values | **0.0%** | expl 7.35% — impractical |
| Frozen warmup (950/1K~950/5K) | Fix early, free later | 2.6~13.2% | Reverts to baseline with more free iterations |
| Root-only QP (2-Stage) | Approximate NE then readjust root | 3.6~20.3% | Subtree mismatch |
| Donk-mass LP | Greedy donk maximization | 2.2% donk | bet EV < check EV for most hands |
| Cum_strategy bias (delta) | Check weight on cumulative strategy | 10.6% | expl 1.08% too high |
| Delta + Beta combination | Both techniques combined | 10.1~12.0% | Worse than beta alone |
| Adaptive eps (gap-based) | Variable epsilon based on EV gap | 37%→32% | Signal too noisy |
| All-node check reward | Check reward at every node | expl 3.03% | Distorts entire tree |
| **Root Check Reward (beta=8.0)** | **Epsilon added to check CF value at root only** | **9.1%** | **Adopted. Current best** |

#### EGT mu Schedule Experiments

| Schedule | Result |
|----------|--------|
| Standard O(1/t^2) mu decay | Diverges at iter 32~64 |
| Geometric decay (mu→0.001) | Diverges at iter 4096~8192 |
| Staircase (20 phases, halving) | Diverges at phase 16~ |
| Average iterate + standard schedule | Divergent iterates contaminate average |
| O(1/t) k=50 | Converges 0.24%@10K (too slow) |
| O(1/t) k=5 | Converges 0.55%@10K (early plateau) |
| **O(1/t) k=20** | **Converges 0.17%@10K (optimal)** |

#### The Winner: Root Check Reward

The Root Check Reward adds a small epsilon to the check action's counterfactual value at the root node only. This nudges indifferent hands toward a more passive equilibrium. At `pref-beta=8.0`, the per-hand difference drops to 9.1% with only 0.19% exploitability.

The remaining 9.1% is the theoretical limit of a uniform single-parameter bias. Per-hand selective equilibrium selection remains an open research problem.

### Phase 6: Preflop Solver

A postflop solver is useless without preflop ranges. We built a 6-max preflop solver using External Sampling MCCFR:

- 2.2 million information sets
- 100 million iterations in 14 minutes
- 122K iterations/second
- Automatic extraction of 30 matchup ranges (SRP, 3-bet, 4-bet pots)

The preflop ranges feed directly into the postflop solver for realistic game scenarios.

### Phase 7: Alternative Algorithms

Beyond standard DCFR, we implemented and validated two alternative equilibrium-finding algorithms:

- **EGT (Excessive Gap Technique)**: Converges to a MaxEnt (maximum entropy) Nash Equilibrium. Uses gradient descent with proximal updates. Produces a unique, deterministic equilibrium. 0.22% exploitability on flop solves.

- **QRE (Quantal Response Equilibrium)**: Fixed-point iteration with softmax strategy computation. Converges to a unique QRE that approximates Nash Equilibrium. Lambda and damping parameters control the entropy-accuracy tradeoff.

Both provide unique equilibria (bit-identical across runs), unlike standard CFR which can vary with implementation details.

## What We Proved

1. **You can build a solver from scratch with near-zero exploitability.** 0.016% exploitability, validated across 8 board textures.
2. **Per-hand frequency differences between solvers are not bugs.** They are different valid Nash Equilibria. Pure strategies (clear bet/check decisions) match 100% across solvers.
3. **Performance is achievable without shortcuts.** +163% optimization through principled engineering.
4. **The isomorphism problem is subtle.** A correct implementation requires careful attention to permutation direction in chance node aggregation. Forward vs inverse permutation matters for non-involutory group elements.

## Features

- **Postflop Solver**: Flop, Turn, and River solving with configurable bet sizes
- **Preflop Solver**: 6-max MCCFR with External Sampling
- **High Accuracy**: 0.016% exploitability at 10K iterations
- **Suit Isomorphism**: Canonical observation mapping with verified S3 group handling
- **Geometric Sizing**: Automatic pot-geometric bet sizing when N bets remain
- **Multiple Algorithms**: DCFR, CFR+, EGT, QRE v2
- **Export**: JSON and HTML strategy visualization

## Build

Requires Rust 1.70+.

```bash
cargo build --release
```

## Usage

### Flop Solve

```bash
dcfr-solver solve --board "Td9d6h" --street flop --oop-range "AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J2s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o" --ip-range "AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J2s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o" --pot 30 --stack 170 --iterations 1000 --bet-sizes 67 --raise-sizes 100 --geometric --output result.html
```

### Turn Solve

```bash
dcfr-solver solve --board "AsKs3d7c" --street turn --oop-range "AA,KK,QQ" --ip-range "JJ,TT,99" --pot 2000 --stack 4000 --iterations 2000 --output result.json
```

### River Solve

```bash
dcfr-solver solve --board "AsKs3d7c2h" --street river --oop-range "AA,KK,QQ" --ip-range "JJ,TT,99" --pot 2000 --stack 4000 --iterations 5000 --output result.html
```

### Preflop Solve

```bash
dcfr-solver preflop --iterations 100000000 --output blueprint.bin
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--bet-sizes` | Bet sizes as % of pot (comma-separated) | `67` |
| `--raise-sizes` | Raise sizes as % of pot | `100` |
| `--geometric` | Use geometric sizing when 2 bets remain | off |
| `--max-raises` | Maximum raises per street | `4` |
| `--allin-pot-ratio` | Min pot ratio for separate all-in action | `0.0` |
| `--pref-beta` | Root check reward for passive NE selection | `0.0` |
| `--no-dcfr` | Use CFR+ instead of DCFR | off |
| `--threads` | Number of threads | all cores |

## Algorithm

The solver uses **DCFR** (Discounted CFR) by default:
- Alpha = 1.5 (positive regret discount: t^alpha / (t^alpha + 1))
- Beta = 0 (negative regret discount: t^beta / (t^beta + 1))
- Gamma = 2.0 (strategy discount: (t / (t+1))^gamma)

Key implementation details:
- Thread-local buffer pools (heap-allocated, prevents stack overflow on deep trees)
- Showdown result caching per river board
- Suit isomorphism with correct S3 permutation group handling
- Parallel chance node evaluation via rayon
- SoA (Struct-of-Arrays) memory layout for SIMD auto-vectorization
- Range-aware compact combo mapping (skip dead + zero-weight combos)
- Regret-Based Pruning for large trees

## Architecture

| File | Description |
|------|-------------|
| `src/cfr.rs` | CFR+/DCFR engine — the hot path (~2,500 lines) |
| `src/game.rs` | Game tree: actions, bet sizing, geometric sizing, streets |
| `src/card.rs` | Card representation (u8), Hand (u64 bitmask) |
| `src/eval.rs` | 7-card hand evaluator |
| `src/range.rs` | Range parser (weighted combos, GTO+ compatible syntax) |
| `src/iso.rs` | Suit isomorphism (canonical observation, S3 permutations) |
| `src/export.rs` | JSON + HTML strategy export |
| `src/preflop.rs` | 6-max preflop MCCFR solver |
| `src/egt.rs` | Excessive Gap Technique (MaxEnt NE) |
| `src/qre.rs` | Quantal Response Equilibrium (unique fixed-point NE) |

## Testing

```bash
cargo test --release
cargo test --release -- --ignored    # Long-running validation tests
```

## Contributing

Pull requests are welcome. If you find a bug, have an optimization idea, or want to add a feature, feel free to open an issue or submit a PR.

Areas where contributions would be especially valuable:
- Performance optimizations (new SIMD strategies, cache-friendly layouts)
- Additional bet sizing configurations and validation
- Alternative CFR variants or equilibrium selection methods
- Multi-threading improvements
- Documentation and examples

## License

MIT
