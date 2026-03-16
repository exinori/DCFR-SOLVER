/// Training data generation for neural network depth-limited solving.
///
/// Generates (input, EV) pairs by solving random turn/river subgames
/// with the existing CFR solver. Each sample captures the equilibrium
/// per-combo utilities for a given (board, ranges, pot, stacks) state.

use crate::card::{combo_from_index, rank, suit, Card, Hand, NUM_COMBOS};
use crate::eval::{evaluate, Strength};
use crate::iso::{canonical_board, combo_permutation};
use crate::cfr::{DcfrMode, SubgameConfig, SubgameSolver};
use crate::game::{BetConfig, BetSize, Street};
use crate::range::Range;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::fs;
use std::io::{BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use rayon::prelude::*;

/// Binary format magic bytes.
const MAGIC: [u8; 4] = [b'P', b'G', b'S', b'1'];
const FORMAT_VERSION: u32 = 1;

/// One training sample: solved subgame with per-combo EVs.
pub struct TrainingSample {
    pub board: Vec<Card>,
    pub street: Street,
    pub pot: i32,
    pub stacks: [i32; 2],
    pub ranges: [Range; 2],
    /// Per-combo utility under equilibrium strategy (reach-weighted).
    /// ev[0] = OOP, ev[1] = IP. Indexed by original combo index.
    pub ev: [[f32; NUM_COMBOS]; 2],
    pub iterations: u32,
    pub exploitability_pct: f32,
}

/// Configuration for data generation.
pub struct DatagenConfig {
    pub count: usize,
    pub street: Street,
    pub iterations: u32,
    pub output_dir: String,
    pub seed: u64,
    pub matchups_path: Option<String>,
    pub min_spr: f32,
    pub max_spr: f32,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        DatagenConfig {
            count: 10000,
            street: Street::River,
            iterations: 300,
            output_dir: "training_data".to_string(),
            seed: 42,
            matchups_path: None,
            min_spr: 0.5,
            max_spr: 20.0,
        }
    }
}

/// Generate random board cards (no duplicates).
fn random_board(rng: &mut SmallRng, n_cards: usize) -> (Hand, Vec<Card>) {
    let mut board = Hand::new();
    let mut cards = Vec::with_capacity(n_cards);
    while cards.len() < n_cards {
        let c = rng.gen_range(0..52u8);
        if !board.contains(c) {
            board = board.add(c);
            cards.push(c);
        }
    }
    (board, cards)
}

/// Generate random pot and stacks with SPR in [min_spr, max_spr].
/// Pot uses log-uniform distribution over [10, 2000] for good coverage
/// across SRP (small pot), 3bet, 4bet, and deep scenarios.
fn random_pot_stacks(rng: &mut SmallRng, min_spr: f32, max_spr: f32) -> (i32, [i32; 2]) {
    // Log-uniform over [10, 2000]: equal probability per order of magnitude
    let log_min = 10.0f32.ln();
    let log_max = 2000.0f32.ln();
    let pot = rng.gen_range(log_min..=log_max).exp() as i32;
    let pot = pot.max(10);
    let spr = rng.gen_range(min_spr..=max_spr);
    let stack = ((pot as f32 * spr) as i32).max(1);
    (pot, [stack, stack])
}

/// Randomly thin a range to simulate postflop reach evolution.
///
/// During actual postflop play, many combos get folded or take different
/// action lines, reducing effective range density from ~50-80% (preflop)
/// to ~3-15% (deep in the game tree). This function simulates that by
/// randomly dropping combos and scaling remaining weights.
///
/// `density_ratio`: fraction of active combos to keep (e.g., 0.1 = keep 10%)
fn subset_range(range: &mut Range, rng: &mut SmallRng, density_ratio: f32) {
    if density_ratio >= 1.0 {
        return;
    }
    for c in 0..NUM_COMBOS {
        if range.weights[c] > 0.0 {
            if rng.gen::<f32>() > density_ratio {
                // Drop this combo (simulates fold/different action)
                range.weights[c] = 0.0;
            } else {
                // 30% chance: keep weight as-is (covers weight=1.0 at inference)
                // 70% chance: scale randomly to simulate partial reach
                if rng.gen::<f32>() > 0.3 {
                    range.weights[c] *= rng.gen_range(0.1f32..1.0);
                }
            }
        }
    }
}

/// Load matchup ranges from JSON (preflop v2 extract) and convert to Range objects.
fn load_matchup_ranges(path: &str) -> Vec<([Range; 2], String)> {
    use crate::preflop::SrpMatchup;

    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read matchups file {}: {}", path, e));
    let matchups: Vec<SrpMatchup> = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Invalid matchups JSON: {}", e));

    let mut result = Vec::new();
    for m in &matchups {
        let oop_str = range_map_to_string(&m.opener.range);
        let ip_str = range_map_to_string(&m.caller.range);

        let oop_range = Range::parse(&oop_str)
            .unwrap_or_else(|| panic!("Cannot parse OOP range for {}", m.matchup));
        let ip_range = Range::parse(&ip_str)
            .unwrap_or_else(|| panic!("Cannot parse IP range for {}", m.matchup));

        result.push(([oop_range, ip_range], m.matchup.clone()));
    }
    result
}

/// Convert BTreeMap<String, f32> to "hand:weight,..." range string.
fn range_map_to_string(map: &std::collections::BTreeMap<String, f32>) -> String {
    map.iter()
        .map(|(hand, weight)| {
            if (*weight - 1.0).abs() < 0.001 {
                hand.clone()
            } else {
                format!("{}:{:.4}", hand, weight)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Generate random ranges when no matchups are provided.
fn random_uniform_ranges(rng: &mut SmallRng, board: Hand) -> [Range; 2] {
    let mut ranges = [Range::empty(), Range::empty()];
    // Each player gets a random subset with random weights
    let density = rng.gen_range(0.3f32..0.9);
    for p in 0..2 {
        for c in 0..NUM_COMBOS {
            let (c1, c2) = combo_from_index(c as u16);
            if board.contains(c1) || board.contains(c2) {
                continue;
            }
            if rng.gen::<f32>() < density {
                // 30% chance weight=1.0 (matches DL-solve inference distribution)
                // 70% chance random weight in [0.1, 1.0)
                if rng.gen::<f32>() < 0.3 {
                    ranges[p].weights[c] = 1.0;
                } else {
                    ranges[p].weights[c] = rng.gen_range(0.1..1.0);
                }
            }
        }
    }
    ranges
}

/// Config A bet config (67% pot bet, 100% pot raise) — fast, simple.
fn config_a_bet_config() -> Arc<BetConfig> {
    let street_sizes = vec![
        vec![BetSize::Frac(2, 3)],
        vec![BetSize::Frac(1, 1)],
    ];
    Arc::new(BetConfig {
        sizes: [vec![], street_sizes.clone(), street_sizes.clone(), street_sizes],
        max_raises: 2,
        allin_threshold: 0.67,
        allin_pot_ratio: 0.0,
        no_donk: false,
        geometric_2bets: false,
    })
}

/// Solve one subgame and extract per-combo EVs.
fn solve_subgame(
    board: Hand,
    board_cards: &[Card],
    street: Street,
    pot: i32,
    stacks: [i32; 2],
    ranges: [Range; 2],
    iterations: u32,
) -> TrainingSample {
    let config = SubgameConfig {
        board,
        pot,
        stacks,
        ranges: ranges.clone(),
        iterations,
        street,
        warmup_frac: 0.0,
        bet_config: Some(config_a_bet_config()),
        dcfr: true,
        cfr_plus: true,
        skip_cum_strategy: true,
        dcfr_mode: DcfrMode::Standard,
        depth_limit: None,
        rake_pct: 0.0,
        rake_cap: 0.0,
        exploration_eps: 0.0,
        entropy_bonus: 0.0, opp_dilute: 0.0, softmax_temp: 0.0,
    };

    let mut solver = SubgameSolver::new(config);
    solver.solve();

    let expl = solver.exploitability_pct();
    let ev = solver.root_ev_per_combo();

    TrainingSample {
        board: board_cards.to_vec(),
        street,
        pot,
        stacks,
        ranges,
        ev,
        iterations,
        exploitability_pct: expl,
    }
}

/// Write training samples to binary file.
///
/// Format (little-endian):
///   magic: [u8; 4] = "PGS1"
///   version: u32
///   n_samples: u32
///   per sample:
///     street: u8 (2=turn, 3=river)
///     board: [u8; 5] (0xFF for unused slots)
///     pot: i32
///     stacks: [i32; 2]
///     ranges: [f32; 1326 * 2] (OOP then IP)
///     ev: [f32; 1326 * 2] (OOP then IP)
///     iterations: u32
///     exploitability_pct: f32
pub fn write_samples(path: &str, samples: &[TrainingSample]) -> std::io::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(&MAGIC)?;
    w.write_u32::<LittleEndian>(FORMAT_VERSION)?;
    w.write_u32::<LittleEndian>(samples.len() as u32)?;

    for s in samples {
        w.write_u8(s.street.index() as u8)?;

        // Board cards (padded to 5 with 0xFF)
        for i in 0..5 {
            if i < s.board.len() {
                w.write_u8(s.board[i])?;
            } else {
                w.write_u8(0xFF)?;
            }
        }

        w.write_i32::<LittleEndian>(s.pot)?;
        w.write_i32::<LittleEndian>(s.stacks[0])?;
        w.write_i32::<LittleEndian>(s.stacks[1])?;

        // Ranges (OOP then IP)
        for p in 0..2 {
            for c in 0..NUM_COMBOS {
                w.write_f32::<LittleEndian>(s.ranges[p].weights[c])?;
            }
        }

        // EVs (OOP then IP)
        for p in 0..2 {
            for c in 0..NUM_COMBOS {
                w.write_f32::<LittleEndian>(s.ev[p][c])?;
            }
        }

        w.write_u32::<LittleEndian>(s.iterations)?;
        w.write_f32::<LittleEndian>(s.exploitability_pct)?;
    }

    w.flush()?;
    Ok(())
}

/// Read training samples from binary file.
pub fn read_samples(path: &str) -> std::io::Result<Vec<TrainingSample>> {
    let file = fs::File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    assert_eq!(magic, MAGIC, "Invalid file magic");
    let version = r.read_u32::<LittleEndian>()?;
    assert_eq!(version, FORMAT_VERSION, "Unsupported format version");
    let n_samples = r.read_u32::<LittleEndian>()? as usize;

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let street_idx = r.read_u8()?;
        let street = match street_idx {
            2 => Street::Turn,
            3 => Street::River,
            _ => panic!("Invalid street index: {}", street_idx),
        };

        let mut board_raw = [0u8; 5];
        r.read_exact(&mut board_raw)?;
        let n_board = if street == Street::Turn { 4 } else { 5 };
        let board: Vec<Card> = board_raw[..n_board].to_vec();

        let pot = r.read_i32::<LittleEndian>()?;
        let stack0 = r.read_i32::<LittleEndian>()?;
        let stack1 = r.read_i32::<LittleEndian>()?;

        let mut ranges = [Range::empty(), Range::empty()];
        for p in 0..2 {
            for c in 0..NUM_COMBOS {
                ranges[p].weights[c] = r.read_f32::<LittleEndian>()?;
            }
        }

        let mut ev = [[0.0f32; NUM_COMBOS]; 2];
        for p in 0..2 {
            for c in 0..NUM_COMBOS {
                ev[p][c] = r.read_f32::<LittleEndian>()?;
            }
        }

        let iterations = r.read_u32::<LittleEndian>()?;
        let exploitability_pct = r.read_f32::<LittleEndian>()?;

        samples.push(TrainingSample {
            board,
            street,
            pot,
            stacks: [stack0, stack1],
            ranges,
            ev,
            iterations,
            exploitability_pct,
        });
    }

    Ok(samples)
}

/// Main datagen entry point.
pub fn run_datagen(config: DatagenConfig) {
    // 64MB stack per rayon thread to avoid stack overflow in cfr_traverse
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok(); // ignore if already initialized

    let mut rng = SmallRng::seed_from_u64(config.seed);

    let matchups = config.matchups_path.as_ref().map(|p| load_matchup_ranges(p));

    fs::create_dir_all(&config.output_dir)
        .unwrap_or_else(|e| panic!("Cannot create output dir: {}", e));

    let n_board_cards = match config.street {
        Street::Turn => 4,
        Street::River => 5,
        _ => panic!("datagen only supports turn/river"),
    };

    let street_name = match config.street {
        Street::Turn => "turn",
        Street::River => "river",
        _ => unreachable!(),
    };

    println!("Generating {} {} training samples (parallel)...", config.count, street_name);
    println!("  iterations per subgame: {}", config.iterations);
    println!("  SPR range: {:.1} - {:.1}", config.min_spr, config.max_spr);
    if let Some(ref m) = matchups {
        println!("  Using {} matchup ranges", m.len());
    } else {
        println!("  Using random ranges (no matchups provided)");
    }

    // Phase 1: Pre-generate all random params sequentially (fast)
    let mut tasks = Vec::with_capacity(config.count);
    let mut total_skipped = 0usize;

    while tasks.len() < config.count {
        let (board, board_cards) = random_board(&mut rng, n_board_cards);

        let ranges = if let Some(ref matchup_list) = matchups {
            let m_idx = rng.gen_range(0..matchup_list.len());
            let (ref m_ranges, _) = matchup_list[m_idx];
            let mut ranges = m_ranges.clone();
            ranges[0].remove_blockers(board);
            ranges[1].remove_blockers(board);

            // Apply random range subsetting to simulate postflop reach evolution.
            // 50% of samples keep full ranges (covers DL-solve inference distribution),
            // 50% are thinned. Log-uniform density in [0.03, 1.0] covers narrow→wide.
            if rng.gen::<f32>() > 0.5 {
                let log_min = 0.03f32.ln();
                let log_max = 1.0f32.ln();
                let density = rng.gen_range(log_min..=log_max).exp();
                subset_range(&mut ranges[0], &mut rng, density);
                subset_range(&mut ranges[1], &mut rng, density);
            }

            ranges
        } else {
            random_uniform_ranges(&mut rng, board)
        };

        if ranges[0].count_live() < 10 || ranges[1].count_live() < 10 {
            total_skipped += 1;
            continue;
        }

        // Canonicalize board and ranges for suit isomorphism
        let (canon_board, suit_perm) = canonical_board(board);
        let cperm = combo_permutation(&suit_perm);
        let canon_board_cards: Vec<Card> = board_cards.iter().map(|&c| {
            rank(c) * 4 + suit_perm[suit(c) as usize]
        }).collect();
        let mut canon_ranges = [Range::empty(), Range::empty()];
        for c in 0..NUM_COMBOS {
            let cc = cperm[c] as usize;
            canon_ranges[0].weights[cc] = ranges[0].weights[c];
            canon_ranges[1].weights[cc] = ranges[1].weights[c];
        }

        let (pot, stacks) = random_pot_stacks(&mut rng, config.min_spr, config.max_spr);
        tasks.push((canon_board, canon_board_cards, canon_ranges, pot, stacks));
    }

    println!("  Pre-generated {} tasks ({} skipped)", tasks.len(), total_skipped);

    // Phase 2: Solve in batches and write incrementally
    // Instead of solving ALL tasks then writing, process in chunks of batch_size.
    // This produces output files as batches complete (crucial for slow turn solves).
    let start = Instant::now();
    let solved_count = AtomicUsize::new(0);
    let total = tasks.len();
    let street = config.street;
    let iterations = config.iterations;
    let batch_size = 1000;
    let mut total_written = 0usize;

    for (batch_idx, chunk) in tasks.chunks(batch_size).enumerate() {
        let chunk_vec: Vec<_> = chunk.to_vec();
        let samples: Vec<TrainingSample> = chunk_vec
            .into_par_iter()
            .map(|(board, board_cards, ranges, pot, stacks)| {
                let sample = solve_subgame(
                    board, &board_cards, street,
                    pot, stacks, ranges, iterations,
                );

                let done = solved_count.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 || done == 1 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = done as f64 / elapsed;
                    let remaining = (total - done) as f64 / rate;
                    eprintln!(
                        "  [{}/{}] {:.1} samples/sec, ETA {:.0}m",
                        done, total, rate, remaining / 60.0,
                    );
                }

                sample
            })
            .collect();

        let filename = format!("{}_{:06}.bin", street_name, batch_idx);
        let path = Path::new(&config.output_dir).join(&filename);
        write_samples(path.to_str().unwrap(), &samples)
            .unwrap_or_else(|e| panic!("Cannot write {}: {}", filename, e));
        total_written += samples.len();
        eprintln!("  Wrote {} samples to {} (total: {})", samples.len(), filename, total_written);
    }

    // Write metadata
    let metadata = serde_json::json!({
        "street": street_name,
        "count": total_written,
        "skipped": total_skipped,
        "iterations": config.iterations,
        "min_spr": config.min_spr,
        "max_spr": config.max_spr,
        "seed": config.seed,
        "elapsed_secs": start.elapsed().as_secs_f64(),
        "matchups": config.matchups_path,
        "batch_size": batch_size,
    });
    let meta_path = Path::new(&config.output_dir).join("metadata.json");
    fs::write(
        &meta_path,
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .expect("Cannot write metadata");

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n=== Datagen complete ===");
    println!("  Samples: {}", total_written);
    println!("  Skipped: {}", total_skipped);
    println!("  Time: {:.1}s ({:.1}m)", elapsed, elapsed / 60.0);
    println!("  Rate: {:.2} samples/sec", total_written as f64 / elapsed);
    println!("  Output: {}", config.output_dir);
}

// ---------------------------------------------------------------------------
// Equity computation for delta learning
// ---------------------------------------------------------------------------

/// Magic bytes for equity companion files.
const EQUITY_MAGIC: [u8; 4] = [b'P', b'G', b'S', b'E'];

/// Compute equity-based counterfactual values for a turn position.
///
/// For each of 48 possible river cards, computes showdown utility and averages.
/// This gives the same result as `depth_limited_equity` in cfr.rs, but works
/// directly from training sample data without needing ComboMap or the solver.
///
/// Returns equity CFV for both players: [OOP, IP], indexed by original combo index.
/// Values are raw (NOT pot-normalized) — same scale as training sample EVs.
pub fn compute_equity_cfv(
    board_cards: &[Card],
    ranges: &[Range; 2],
    pot: i32,
) -> [[f32; NUM_COMBOS]; 2] {
    let mut equity = [[0.0f32; NUM_COMBOS]; 2];

    // Build board hand
    let mut board = Hand::new();
    for &c in board_cards {
        board = board.add(c);
    }

    if board_cards.len() >= 5 {
        // River (5-card board): direct showdown evaluation, no additional card
        let mut combo_strength: Vec<(usize, Strength)> = Vec::with_capacity(NUM_COMBOS);
        for c in 0..NUM_COMBOS {
            let (c1, c2) = combo_from_index(c as u16);
            if board.contains(c1) || board.contains(c2) {
                continue;
            }
            let hand = board.add(c1).add(c2);
            let strength = evaluate(hand);
            combo_strength.push((c, strength));
        }
        combo_strength.sort_by_key(|&(_, s)| s);

        for traverser in 0..2usize {
            let opp = 1 - traverser;
            showdown_sweep(
                &combo_strength,
                &ranges[traverser].weights,
                &ranges[opp].weights,
                pot as f32,
                &mut equity[traverser],
            );
        }
    } else {
        // Turn (4-card board): average over all possible river cards
        let available: Vec<u8> = (0..52u8).filter(|&c| !board.contains(c)).collect();
        let n_cards = available.len() as f32;
        if n_cards == 0.0 { return equity; }

        for &river_card in &available {
            let river_board = board.add(river_card);

            let mut combo_strength: Vec<(usize, Strength)> = Vec::with_capacity(NUM_COMBOS);
            for c in 0..NUM_COMBOS {
                let (c1, c2) = combo_from_index(c as u16);
                if river_board.contains(c1) || river_board.contains(c2) {
                    continue;
                }
                let hand = river_board.add(c1).add(c2);
                let strength = evaluate(hand);
                combo_strength.push((c, strength));
            }
            combo_strength.sort_by_key(|&(_, s)| s);

            for traverser in 0..2usize {
                let opp = 1 - traverser;
                showdown_sweep(
                    &combo_strength,
                    &ranges[traverser].weights,
                    &ranges[opp].weights,
                    pot as f32,
                    &mut equity[traverser],
                );
            }
        }

        // Average over river cards
        let inv = 1.0 / n_cards;
        for p in 0..2 {
            for c in 0..NUM_COMBOS {
                equity[p][c] *= inv;
            }
        }
    }

    equity
}

/// O(N) showdown CFV sweep over strength-sorted combos.
///
/// Groups combos by hand strength, sweeps from weakest to strongest.
/// Uses per-card inclusion-exclusion for blocker adjustment (same algorithm
/// as `showdown_utility_with_precomputed` in cfr.rs).
///
/// At root (no bets): CFV[c] = pot * beat_reach + (pot/2) * tie_reach
fn showdown_sweep(
    sorted_combos: &[(usize, Strength)],
    t_reach: &[f32; NUM_COMBOS],
    opp_reach: &[f32; NUM_COMBOS],
    pot: f32,
    out: &mut [f32; NUM_COMBOS],
) {
    let half_pot = pot * 0.5;
    let n = sorted_combos.len();
    let mut i = 0;

    // Beat accumulators (grows as we sweep from weak to strong)
    let mut total_beat = 0.0f32;
    let mut per_card_beat = [0.0f32; 52];

    while i < n {
        let group_strength = sorted_combos[i].1;
        let group_start = i;

        // Find end of this strength group
        while i < n && sorted_combos[i].1 == group_strength {
            i += 1;
        }

        // Compute tie reach for this group
        let mut total_tie = 0.0f32;
        let mut per_card_tie = [0.0f32; 52];

        for j in group_start..i {
            let (ci, _) = sorted_combos[j];
            let r = opp_reach[ci];
            if r <= 0.0 { continue; }
            total_tie += r;
            let (c1, c2) = combo_from_index(ci as u16);
            per_card_tie[c1 as usize] += r;
            per_card_tie[c2 as usize] += r;
        }

        // Compute CFV for traverser's combos in this group
        for j in group_start..i {
            let (ci, _) = sorted_combos[j];
            if t_reach[ci] <= 0.0 { continue; }

            let (c1, c2) = combo_from_index(ci as u16);
            let c1u = c1 as usize;
            let c2u = c2 as usize;
            let r_overlap = opp_reach[ci];

            let adj_beat = total_beat - per_card_beat[c1u] - per_card_beat[c2u];
            let adj_tie = total_tie - per_card_tie[c1u] - per_card_tie[c2u] + r_overlap;

            out[ci] += pot * adj_beat + half_pot * adj_tie;
        }

        // Transfer tie → beat accumulators
        total_beat += total_tie;
        for j in group_start..i {
            let (ci, _) = sorted_combos[j];
            let r = opp_reach[ci];
            if r <= 0.0 { continue; }
            let (c1, c2) = combo_from_index(ci as u16);
            per_card_beat[c1 as usize] += r;
            per_card_beat[c2 as usize] += r;
        }
    }
}

/// Write equity companion file for a set of samples.
///
/// Format: PGSE magic (4) + n_samples (u32) + per sample: 2×1326 f32
pub fn write_equity_file(path: &str, equity_data: &[[[f32; NUM_COMBOS]; 2]]) -> std::io::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(&EQUITY_MAGIC)?;
    w.write_u32::<LittleEndian>(equity_data.len() as u32)?;

    for eq in equity_data {
        for p in 0..2 {
            for c in 0..NUM_COMBOS {
                w.write_f32::<LittleEndian>(eq[p][c])?;
            }
        }
    }

    w.flush()?;
    Ok(())
}

/// Read equity companion file.
pub fn read_equity_file(path: &str) -> std::io::Result<Vec<[[f32; NUM_COMBOS]; 2]>> {
    let file = fs::File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    assert_eq!(magic, EQUITY_MAGIC, "Invalid equity file magic");
    let n_samples = r.read_u32::<LittleEndian>()? as usize;

    let mut data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut eq = [[0.0f32; NUM_COMBOS]; 2];
        for p in 0..2 {
            for c in 0..NUM_COMBOS {
                eq[p][c] = r.read_f32::<LittleEndian>()?;
            }
        }
        data.push(eq);
    }

    Ok(data)
}

/// Compute equity for all samples in .bin files and write companion .equity.bin files.
///
/// For each input .bin file, reads the samples, computes equity CFV using
/// averaged showdown over river cards, and writes a companion file.
pub fn run_compute_equity(data_dir: &str) {
    // 64MB stack per rayon thread
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    let bin_files: Vec<_> = fs::read_dir(data_dir)
        .expect("Cannot read data dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
        .filter(|e| !e.path().to_str().unwrap_or("").contains(".equity."))
        .map(|e| e.path())
        .collect();

    let mut bin_files = bin_files;
    bin_files.sort();

    println!("Computing equity for {} .bin files in {}", bin_files.len(), data_dir);
    let start = Instant::now();
    let mut total_samples = 0usize;

    for (fi, bin_path) in bin_files.iter().enumerate() {
        let equity_path = bin_path.with_extension("equity.bin");
        if equity_path.exists() {
            println!("  [{}/{}] {} — already exists, skipping",
                fi + 1, bin_files.len(), equity_path.display());
            continue;
        }

        let samples = read_samples(bin_path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("Cannot read {}: {}", bin_path.display(), e));

        println!("  [{}/{}] {} — {} samples, computing equity...",
            fi + 1, bin_files.len(), bin_path.display(), samples.len());

        // Compute equity in parallel
        let equity_data: Vec<[[f32; NUM_COMBOS]; 2]> = samples
            .par_iter()
            .map(|s| compute_equity_cfv(&s.board, &s.ranges, s.pot))
            .collect();

        write_equity_file(equity_path.to_str().unwrap(), &equity_data)
            .unwrap_or_else(|e| panic!("Cannot write {}: {}", equity_path.display(), e));

        total_samples += samples.len();
        let elapsed = start.elapsed().as_secs_f64();
        let rate = total_samples as f64 / elapsed;
        println!("    Done. {:.1} samples/sec total", rate);
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n=== Equity computation complete ===");
    println!("  Samples processed: {}", total_samples);
    println!("  Time: {:.1}s ({:.1}m)", elapsed, elapsed / 60.0);
}
