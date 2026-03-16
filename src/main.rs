use clap::{Parser, Subcommand};
use dcfr_solver::abstraction::EquityAbstraction;
use dcfr_solver::card::{parse_cards, Hand};
use dcfr_solver::cfr::{DcfrMode, SubgameConfig, SubgameSolver};
use dcfr_solver::export::{export_preflop_chart, SolveResult};
use dcfr_solver::game::{BetConfig, BetSize, Street};
use std::sync::Arc;
use dcfr_solver::mccfr::McfrTrainer;
use dcfr_solver::preflop::{PreflopBetConfig, PreflopTrainer};
use dcfr_solver::batch;
use dcfr_solver::range::Range;
use dcfr_solver::strategy::PreflopChart;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "dcfr-solver", about = "GTO poker solver")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Precompute equity abstraction tables
    Abstract {
        #[arg(long, default_value = "abstraction.bin")]
        output: String,
    },

    /// Train blueprint strategy via MCCFR
    Train {
        #[arg(long, default_value_t = 1_000_000)]
        iterations: u64,

        #[arg(long, default_value = "abstraction.bin")]
        abstraction: String,

        #[arg(long, default_value = "blueprint.bin")]
        output: String,

        #[arg(long, default_value_t = 100)]
        stack: i32,

        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Solve a specific postflop subgame
    Solve {
        #[arg(long)]
        board: String,

        #[arg(long)]
        oop_range: String,

        #[arg(long)]
        ip_range: String,

        #[arg(long, default_value_t = 20)]
        pot: i32,

        #[arg(long, default_value_t = 90)]
        stack: i32,

        #[arg(long, default_value = "river")]
        street: String,

        #[arg(long, default_value_t = 1000)]
        iterations: u32,

        /// Bet sizes as pot fractions, e.g., "33,75,150" for 33%, 75%, 150% pot.
        /// Applies to all streets. Default: use built-in grid.
        #[arg(long)]
        bet_sizes: Option<String>,

        /// Raise sizes as pot fractions, e.g., "75,200" for 75%, 200% pot.
        #[arg(long)]
        raise_sizes: Option<String>,

        /// Output format: "json" or "html" (auto-detected from extension if not set)
        #[arg(long)]
        format: Option<String>,

        #[arg(long, default_value = "result.json")]
        output: String,

        /// Skip cum_strategy allocation (use regret-matched strategy).
        /// Saves ~50% memory. Safe with DCFR at 300+ iterations.
        #[arg(long)]
        skip_cum_strategy: bool,

        /// Disable DCFR discounting (use vanilla CFR+).
        #[arg(long)]
        no_dcfr: bool,

        /// All-in pot ratio: only add all-in when stack ≤ pot × this ratio.
        /// 0.0 = always allow all-in (default). 3.0 = GTO+ standard (SPR ≤ 3).
        #[arg(long, default_value_t = 0.0)]
        allin_pot_ratio: f32,

        /// All-in threshold: collapse bet/raise to all-in when it uses ≥ this
        /// fraction of remaining stack. 0.0 = disabled. 0.67 = GTO+ standard.
        #[arg(long, default_value_t = 0.0)]
        allin_threshold: f32,

        /// Max raises per street. Default: 3.
        #[arg(long, default_value_t = 3)]
        max_raises: u8,

        /// Disable OOP donk betting (OOP can only check when first to act).
        /// Matches GTO+ "Donk OFF" setting.
        #[arg(long)]
        no_donk: bool,

        /// Use geometric sizing when only 2 bets remain before cap.
        /// Matches GTO+ "With only 2 bets left, use geometric sizing".
        #[arg(long)]
        geometric: bool,

        /// Exploration epsilon for strategy mixing.
        /// strategy = (1-eps) * RM+(regret) + eps/n_actions.
        /// Encourages natural mixing on indifferent hands. 0.0 = disabled.
        #[arg(long, default_value_t = 0.0)]
        exploration_eps: f32,

        /// Post-processing: smooth indifferent hands toward aggregate frequency.
        /// Threshold in chips: combos with EV gap < threshold get smoothed.
        /// 0.0 = disabled. Try 0.1, 0.3, 0.5, 1.0, 2.0.
        #[arg(long, default_value_t = 0.0)]
        smooth_threshold: f32,

        /// MaxEnt regularization: entropy bonus (τ) added to regrets during CFR.
        /// Encourages mixed strategies. 0.0 = disabled. Try 0.01, 0.1, 0.5.
        #[arg(long, default_value_t = 0.0)]
        entropy_bonus: f32,

        /// Linearly anneal entropy_bonus to 0 over iterations.
        /// High τ early (exploration) → 0 later (convergence to proper NE).
        #[arg(long)]
        entropy_anneal: bool,

        /// Apply entropy bonus only at root decision node (OOP's first action).
        /// Limits exploitability impact while encouraging mixing at the root.
        #[arg(long)]
        entropy_root_only: bool,

        /// Diluted best response: mix opponent strategy with uniform by δ.
        /// σ_opp = (1-δ)*σ_opp + δ*uniform. 0.0 = disabled. Try 0.01, 0.05, 0.1.
        #[arg(long, default_value_t = 0.0)]
        opp_dilute: f32,

        /// Softmax temperature for FTRL-style regret matching (MaxEnt NE).
        /// Replaces RM+ with softmax: σ(a) = exp(R(a)/τ) / Σexp(R(a)/τ).
        /// 0.0 = disabled (standard RM+). Try 1.0, 5.0, 10.0.
        #[arg(long, default_value_t = 0.0)]
        softmax_temp: f32,

        /// Use LCFR (Linear CFR) discounting instead of standard DCFR.
        /// Linear weights (alpha=1, gamma=1) may converge to a different NE.
        #[arg(long)]
        lcfr: bool,

        /// Algorithm: "cfr" (default), "egt", "qre" (softmax CFR), or "qre2" (QRE fixed-point).
        #[arg(long, default_value = "cfr")]
        algorithm: String,

        /// QRE2 precision parameter λ. Higher = more rational (closer to NE).
        /// With pot=2000: λ=0.001 → smooth QRE, λ=0.01 → moderate, λ=0.1 → near-NE.
        #[arg(long, default_value_t = 0.01)]
        qre_lambda: f32,

        /// QRE2 damping: mixing rate with new strategy per iteration.
        /// 1.0 = full update, 0.5 = half old + half new (more stable).
        #[arg(long, default_value_t = 0.5)]
        qre_damping: f32,

        /// QRE2 lambda annealing: ramp λ from λ/100 to λ over iterations.
        #[arg(long)]
        qre_anneal: bool,

        /// Purify: zero out actions below this percentage and renormalize.
        /// E.g., --purify 5 removes all actions played <5%. Applied after solve.
        #[arg(long, default_value_t = 0.0)]
        purify: f32,

        /// Passive tie-break: push indifferent combos toward check/call.
        /// Value is EV threshold in chips. Combos with EV gap < threshold get pushed.
        /// E.g., --passive-tiebreak 0.5
        #[arg(long, default_value_t = 0.0)]
        passive_tiebreak: f32,

        /// Per-street bet sizes (override --bet-sizes for specific street).
        /// E.g., --flop-bet "67" --turn-bet "33,67,125" --river-bet "33,67,125"
        #[arg(long)]
        flop_bet: Option<String>,
        #[arg(long)]
        flop_raise: Option<String>,
        #[arg(long)]
        turn_bet: Option<String>,
        #[arg(long)]
        turn_raise: Option<String>,
        #[arg(long)]
        river_bet: Option<String>,
        #[arg(long)]
        river_raise: Option<String>,

        /// Rake percentage (0.05 = 5%). Default: 0 (no rake).
        #[arg(long, default_value_t = 0.0)]
        rake: f32,

        /// Rake cap in chips. Default: 0 (no cap).
        #[arg(long, default_value_t = 0.0)]
        rake_cap: f32,

        /// Depth limit: stop at this street and use NN evaluation.
        /// E.g., --depth-limit river stops at river and uses NN.
        #[arg(long)]
        depth_limit: Option<String>,

        /// Path to ONNX value network model for depth-limited solving.
        #[arg(long)]
        valuenet: Option<String>,

        /// Disable suit isomorphism at chance nodes (debug: slower but removes ISO as variable).
        #[arg(long)]
        no_iso: bool,

        /// Regret matching floor: adds decaying positive offset to regrets before CFR+ clamping.
        /// Prevents early clamping from permanently killing actions (e.g., donk bets).
        /// Floor decays as rm_floor/(1+0.01*iter). 0.0 = disabled. Try 1.0, 5.0, 10.0.
        #[arg(long, default_value_t = 0.0)]
        rm_floor: f32,

        /// Alternating updates: only traverse one player per iteration (OOP odd, IP even).
        /// Matches Tammelin's original CFR+ paper.
        #[arg(long)]
        alternating: bool,

        /// Linear t-weighting on cum_strategy: multiply contribution by iteration t.
        /// Standard CFR+ averaging (later iterations contribute more).
        #[arg(long)]
        t_weight: bool,

        /// Frozen root: load GTO+ strategy file to pin root OOP strategy.
        /// File format: GTO+ tab-separated (hand at col 0, bet% at col 9).
        #[arg(long)]
        frozen_root: Option<String>,

        /// Two-phase solve: solve normally, then re-solve with indifferent hands
        /// set to 50/50 as frozen root. Value = EV gap threshold in chips.
        /// 0.0 = disabled. Try 0.1, 0.5, 1.0.
        #[arg(long, default_value_t = 0.0)]
        two_phase: f32,

        /// Seed root check regret to bias toward passive NE.
        /// 0.0 = disabled. Try 0.1, 1.0, 10.0.
        #[arg(long, default_value_t = 0.0)]
        check_bias: f32,

        /// Override DCFR gamma (cum_strategy discounting). -1 = use mode default.
        /// 0=uniform avg, 1=linear, 2=standard DCFR. Try 0, 0.5, 1, 3.
        #[arg(long, default_value_t = -1.0)]
        dcfr_gamma: f64,

        /// Override DCFR alpha (positive regret discounting). -1 = use mode default.
        /// 0=no discount, 1=linear, 1.5=standard DCFR. Requires --dcfr-gamma.
        #[arg(long, default_value_t = -1.0)]
        dcfr_alpha: f64,

        /// Preference-CFR delta for passive actions (check/fold).
        /// Multiplies action 0 regret in RM step to bias toward passive NE.
        /// 1.0 = disabled. Try 2.0, 3.0, 5.0.
        #[arg(long, default_value_t = 1.0)]
        pref_delta: f32,

        /// Root check reward: pot-relative epsilon for passive NE selection.
        /// Adds pref_beta * pot to check action's utility. Try 7.0-8.0.
        #[arg(long, default_value_t = 0.0)]
        pref_beta: f32,

        /// Apply check reward to all nodes (not just root).
        #[arg(long, default_value_t = false)]
        pref_beta_all: bool,

        /// Regret-Based Pruning: skip zero-strategy subtrees at traverser nodes.
        /// Significant speedup for large trees (3-bet+). Default false.
        #[arg(long, default_value_t = false)]
        pruning: bool,

        /// Per-combo check bias file (JSON array of 1326 floats, indexed by original combo).
        /// Positive = encourage check, negative = encourage bet.
        #[arg(long)]
        combo_bias: Option<String>,

        /// Frozen root warm-up: run first N iterations with frozen_root, then unfreeze.
        /// Only effective when --frozen-root is also set.
        #[arg(long, default_value_t = 0)]
        frozen_warmup: u32,

        /// Decay factor for root cum_strategy at unfreeze time. 1.0 = keep all (default).
        /// 0.0 = full reset, 0.1 = keep 10%.
        #[arg(long, default_value_t = 1.0)]
        unfreeze_decay: f32,
    },

    /// Extract preflop chart from blueprint
    Chart {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,

        #[arg(long, default_value = "chart.json")]
        output: String,
    },

    /// Train 6-max preflop GTO strategy via MCCFR
    Preflop {
        #[arg(long, default_value_t = 10_000_000)]
        iterations: u64,

        #[arg(long, default_value = "preflop_blueprint.bin")]
        output: String,

        /// JSON chart output (optional)
        #[arg(long)]
        chart_output: Option<String>,

        /// SRP matchup ranges output (optional)
        #[arg(long)]
        matchup_output: Option<String>,

        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Open raise size in chips (1 chip = 0.5bb). Default: 5 = 2.5bb (NL500)
        #[arg(long, default_value_t = 5)]
        open_size: i32,

        /// SB open size in chips. Default: 7 = 3.5bb (NL500)
        #[arg(long, default_value_t = 7)]
        sb_open_size: i32,

        /// 3-bet size in chips. Default: 18 = 9bb (NL500)
        #[arg(long, default_value_t = 18)]
        bet3_size: i32,

        /// 4-bet size in chips. Default: 44 = 22bb
        #[arg(long, default_value_t = 44)]
        bet4_size: i32,

        /// Allow SB to limp (call instead of fold/raise)
        #[arg(long, default_value_t = true)]
        sb_limp: bool,

        /// OOP position tax as fraction of pot (0.0-0.5). Default: 0.20
        /// Transfers this fraction of pot × (rank gap/5) from OOP to IP at showdown.
        /// Accounts for positional disadvantage. Set to 0.0 for raw equity.
        #[arg(long, default_value_t = 0.20)]
        oop_pot_tax: f32,
    },

    /// Extract RFI charts from a preflop blueprint
    ChartPreflop {
        #[arg(long, default_value = "preflop_blueprint.bin")]
        blueprint: String,

        #[arg(long, default_value = "preflop_charts.json")]
        output: String,

        /// SRP matchup ranges output (optional)
        #[arg(long)]
        matchup_output: Option<String>,
    },

    /// Generate batch solve configs (JSONL) from preflop matchup ranges
    BatchConfig {
        /// Path to matchups.json (from chart-preflop --matchup-output)
        #[arg(long)]
        matchups: String,

        /// Generate defender range template JSON (write to this path and exit)
        #[arg(long)]
        generate_template: Option<String>,

        /// Path to defender range overrides JSON
        #[arg(long)]
        defender_ranges: Option<String>,

        /// Postflop solver iterations per spot
        #[arg(long, default_value_t = 350)]
        iterations: u32,

        /// Output JSONL path
        #[arg(long, default_value = "batch_configs.jsonl")]
        output: String,
    },

    /// Run batch solver: read JSONL configs, solve each spot, save JSON results.
    /// Default bet config: Config D (33%+67%+125% bet, 50%+100% raise, allin≤3×pot).
    BatchRun {
        /// Input JSONL config file (from batch-config)
        #[arg(long, default_value = "batch_configs.jsonl")]
        input: String,

        /// Output directory for result JSON files
        #[arg(long, default_value = "results")]
        output_dir: String,

        /// Bet sizes as pot fractions, e.g., "33,67,125".
        /// Default: 33,67,125 (Config D)
        #[arg(long)]
        bet_sizes: Option<String>,

        /// Raise sizes as pot fractions, e.g., "50,100".
        /// Default: 50,100 (Config D)
        #[arg(long)]
        raise_sizes: Option<String>,

        /// Start line (0-indexed, inclusive). For splitting across instances.
        #[arg(long, default_value_t = 0)]
        start: usize,

        /// Number of spots to solve (0 = all remaining)
        #[arg(long, default_value_t = 0)]
        count: usize,

        /// Skip spots that already have result files
        #[arg(long, default_value_t = true)]
        skip_existing: bool,

        /// Skip cum_strategy allocation (use regret-matched strategy).
        /// Saves ~50% memory. Safe with DCFR at 300+ iterations.
        #[arg(long, default_value_t = true)]
        skip_cum_strategy: bool,
    },

    /// Generate training data for NN depth-limited solving
    #[cfg(feature = "nn")]
    Datagen {
        /// Street to solve: "turn" or "river"
        #[arg(long, default_value = "river")]
        street: String,

        /// Number of samples to generate
        #[arg(long, default_value_t = 10000)]
        count: usize,

        /// CFR iterations per subgame
        #[arg(long, default_value_t = 300)]
        iterations: u32,

        /// Output directory for .bin files
        #[arg(long, default_value = "training_data")]
        output_dir: String,

        /// Path to matchup ranges JSON (from preflop chart-preflop --matchup-output)
        #[arg(long)]
        matchups: Option<String>,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Minimum stack-to-pot ratio
        #[arg(long, default_value_t = 0.5)]
        min_spr: f32,

        /// Maximum stack-to-pot ratio
        #[arg(long, default_value_t = 20.0)]
        max_spr: f32,
    },

    /// Compute equity companion files for delta learning
    #[cfg(feature = "nn")]
    ComputeEquity {
        /// Directory containing .bin training data files
        #[arg(long, default_value = "training_data")]
        data_dir: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Abstract { output } => {
            cmd_abstract(&output);
        }
        Commands::Train { iterations, abstraction, output, stack, seed } => {
            cmd_train(iterations, &abstraction, &output, stack, seed);
        }
        Commands::Solve { board, oop_range, ip_range, pot, stack, street, iterations, bet_sizes, raise_sizes, format, output, skip_cum_strategy, no_dcfr, depth_limit, valuenet, allin_pot_ratio, allin_threshold, max_raises, no_donk, geometric, exploration_eps, smooth_threshold, entropy_bonus, entropy_anneal, entropy_root_only, opp_dilute, softmax_temp, lcfr, algorithm, purify, passive_tiebreak, flop_bet, flop_raise, turn_bet, turn_raise, river_bet, river_raise, rake, rake_cap, qre_lambda, qre_damping, qre_anneal, no_iso, rm_floor, alternating, t_weight, frozen_root, two_phase, check_bias, dcfr_gamma, dcfr_alpha, pref_delta, pref_beta, pref_beta_all, pruning, combo_bias, frozen_warmup, unfreeze_decay } => {
            let per_street = PerStreetSizes {
                flop_bet: flop_bet.as_deref(),
                flop_raise: flop_raise.as_deref(),
                turn_bet: turn_bet.as_deref(),
                turn_raise: turn_raise.as_deref(),
                river_bet: river_bet.as_deref(),
                river_raise: river_raise.as_deref(),
            };
            cmd_solve(&board, &oop_range, &ip_range, pot, stack, &street, iterations, bet_sizes.as_deref(), raise_sizes.as_deref(), format.as_deref(), &output, skip_cum_strategy, no_dcfr, depth_limit.as_deref(), valuenet.as_deref(), allin_pot_ratio, allin_threshold, max_raises, no_donk, geometric, exploration_eps, smooth_threshold, entropy_bonus, entropy_anneal, entropy_root_only, opp_dilute, softmax_temp, lcfr, &algorithm, purify, passive_tiebreak, &per_street, rake, rake_cap, qre_lambda, qre_damping, qre_anneal, no_iso, rm_floor, alternating, t_weight, frozen_root.as_deref(), two_phase, check_bias, dcfr_gamma, dcfr_alpha, pref_delta, pref_beta, pref_beta_all, pruning, combo_bias.as_deref(), frozen_warmup, unfreeze_decay);
        }
        Commands::Chart { blueprint, output } => {
            cmd_chart(&blueprint, &output);
        }
        Commands::Preflop { iterations, output, chart_output, matchup_output, seed, open_size, sb_open_size, bet3_size, bet4_size, sb_limp, oop_pot_tax } => {
            cmd_preflop(iterations, &output, chart_output.as_deref(), matchup_output.as_deref(), seed, open_size, sb_open_size, bet3_size, bet4_size, sb_limp, oop_pot_tax);
        }
        Commands::ChartPreflop { blueprint, output, matchup_output } => {
            cmd_chart_preflop(&blueprint, &output, matchup_output.as_deref());
        }
        Commands::BatchConfig { matchups, generate_template, defender_ranges, iterations, output } => {
            cmd_batch_config(&matchups, generate_template.as_deref(), defender_ranges.as_deref(), iterations, &output);
        }
        Commands::BatchRun { input, output_dir, bet_sizes, raise_sizes, start, count, skip_existing, skip_cum_strategy } => {
            cmd_batch_run(&input, &output_dir, bet_sizes.as_deref(), raise_sizes.as_deref(), start, count, skip_existing, skip_cum_strategy);
        }
        #[cfg(feature = "nn")]
        Commands::Datagen { street, count, iterations, output_dir, matchups, seed, min_spr, max_spr } => {
            cmd_datagen(&street, count, iterations, &output_dir, matchups.as_deref(), seed, min_spr, max_spr);
        }
        #[cfg(feature = "nn")]
        Commands::ComputeEquity { data_dir } => {
            dcfr_solver::datagen::run_compute_equity(&data_dir);
        }
    }
}

fn cmd_abstract(output: &str) {
    println!("Building equity abstraction...");
    let abstraction = EquityAbstraction::build();

    let file = fs::File::create(output).expect("cannot create output file");
    let mut writer = BufWriter::new(file);
    abstraction.save(&mut writer).expect("failed to save abstraction");
    println!("Saved to {}", output);
}

fn cmd_train(iterations: u64, abstraction_path: &str, output: &str, stack: i32, seed: u64) {
    println!("Loading abstraction from {}...", abstraction_path);
    let file = fs::File::open(abstraction_path).expect("cannot open abstraction file");
    let mut reader = BufReader::new(file);
    let abstraction = EquityAbstraction::load(&mut reader).expect("failed to load abstraction");

    println!("Training MCCFR for {} iterations (stack={})...", iterations, stack);
    let mut trainer = McfrTrainer::new_with_seed(abstraction, seed);
    trainer.train(iterations, stack);

    println!("Saving blueprint to {}...", output);
    let file = fs::File::create(output).expect("cannot create output file");
    let mut writer = BufWriter::new(file);
    trainer.blueprint.save(&mut writer).expect("failed to save blueprint");
    println!("Done. {} info sets stored.", trainer.blueprint.entries.len());
}

struct PerStreetSizes<'a> {
    flop_bet: Option<&'a str>,
    flop_raise: Option<&'a str>,
    turn_bet: Option<&'a str>,
    turn_raise: Option<&'a str>,
    river_bet: Option<&'a str>,
    river_raise: Option<&'a str>,
}

/// Parse GTO+ tab-separated strategy file into frozen_root array.
/// Format: col[0] = hand (e.g. "AsAh"), col[9] = bet%.
/// Returns Box<[f32; 1326]> indexed by original combo index, storing bet fraction.
fn parse_gtoplus_frozen_root(path: &str) -> Box<[f32; 1326]> {
    use dcfr_solver::card::{parse_card, combo_from_index, NUM_COMBOS};

    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read frozen root file {}: {}", path, e));

    let mut frozen = Box::new([0.5f32; NUM_COMBOS]); // default 50/50

    // Build reverse lookup: (card1, card2) → combo_index
    let mut card_to_combo = std::collections::HashMap::new();
    for idx in 0..NUM_COMBOS {
        let (c1, c2) = combo_from_index(idx as u16);
        card_to_combo.insert((c1, c2), idx);
        card_to_combo.insert((c2, c1), idx);
    }

    let mut matched = 0;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("Hand") { continue; }
        if line.starts_with("pot:") || line.starts_with("stack:") || line.starts_with("to call:") { break; }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 10 { continue; }
        let hand_str = parts[0].trim();
        if hand_str.len() != 4 { continue; }
        let bet_pct: f32 = match parts[9].replace(",", ".").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let c1 = match parse_card(&hand_str[0..2]) { Some(c) => c, None => continue };
        let c2 = match parse_card(&hand_str[2..4]) { Some(c) => c, None => continue };
        if let Some(&idx) = card_to_combo.get(&(c1, c2)) {
            frozen[idx] = bet_pct / 100.0;
            matched += 1;
        }
    }
    println!("  Parsed {} hands from GTO+ file", matched);
    frozen
}

fn cmd_solve(
    board_str: &str,
    oop_range_str: &str,
    ip_range_str: &str,
    pot: i32,
    stack: i32,
    street_str: &str,
    iterations: u32,
    bet_sizes_str: Option<&str>,
    raise_sizes_str: Option<&str>,
    format_str: Option<&str>,
    output: &str,
    skip_cum_strategy: bool,
    no_dcfr: bool,
    depth_limit_str: Option<&str>,
    _valuenet_path: Option<&str>,
    allin_pot_ratio: f32,
    allin_threshold: f32,
    max_raises: u8,
    no_donk: bool,
    geometric: bool,
    exploration_eps: f32,
    smooth_threshold: f32,
    entropy_bonus: f32,
    entropy_anneal: bool,
    entropy_root_only: bool,
    opp_dilute: f32,
    softmax_temp: f32,
    lcfr: bool,
    algorithm: &str,
    purify_pct: f32,
    passive_tiebreak: f32,
    per_street: &PerStreetSizes,
    rake_pct: f32,
    rake_cap: f32,
    qre_lambda: f32,
    qre_damping: f32,
    qre_anneal: bool,
    no_iso: bool,
    rm_floor: f32,
    alternating: bool,
    t_weight: bool,
    frozen_root_file: Option<&str>,
    two_phase: f32,
    check_bias: f32,
    dcfr_gamma: f64,
    dcfr_alpha: f64,
    pref_delta: f32,
    pref_beta: f32,
    pref_beta_all: bool,
    pruning: bool,
    combo_bias_file: Option<&str>,
    frozen_warmup: u32,
    unfreeze_decay: f32,
) {
    let board_cards = parse_cards(board_str).expect("invalid board cards");
    let mut board = Hand::new();
    for c in &board_cards {
        board = board.add(*c);
    }

    let street = match street_str.to_lowercase().as_str() {
        "flop" => {
            assert_eq!(board_cards.len(), 3, "flop needs 3 board cards");
            Street::Flop
        }
        "turn" => {
            assert_eq!(board_cards.len(), 4, "turn needs 4 board cards");
            Street::Turn
        }
        "river" => {
            assert_eq!(board_cards.len(), 5, "river needs 5 board cards");
            Street::River
        }
        _ => panic!("invalid street: {}", street_str),
    };

    let depth_limit = depth_limit_str.map(|s| match s.to_lowercase().as_str() {
        "turn" => Street::Turn,
        "river" => Street::River,
        _ => panic!("invalid depth-limit street: {} (use 'turn' or 'river')", s),
    });

    let oop_range = Range::parse(oop_range_str).expect("invalid OOP range");
    let ip_range = Range::parse(ip_range_str).expect("invalid IP range");

    // Parse custom bet config if provided
    let has_per_street = per_street.flop_bet.is_some() || per_street.flop_raise.is_some()
        || per_street.turn_bet.is_some() || per_street.turn_raise.is_some()
        || per_street.river_bet.is_some() || per_street.river_raise.is_some();

    let bet_config = if has_per_street || bet_sizes_str.is_some() || raise_sizes_str.is_some() {
        // Default fallback from --bet-sizes / --raise-sizes (applies to all streets)
        let default_bet = parse_size_list(bet_sizes_str);
        let default_raise = parse_size_list(raise_sizes_str);

        // Build per-street sizes: per-street option overrides fallback
        let build_street = |bet_opt: Option<&str>, raise_opt: Option<&str>| -> Vec<Vec<BetSize>> {
            let bets = if let Some(s) = bet_opt {
                parse_size_list(Some(s))
            } else {
                default_bet.clone()
            };
            let raises = if let Some(s) = raise_opt {
                parse_size_list(Some(s))
            } else {
                default_raise.clone()
            };
            let mut depths = vec![];
            if !bets.is_empty() {
                depths.push(bets);
            }
            if !raises.is_empty() {
                if depths.is_empty() {
                    depths.push(vec![]); // placeholder for depth 0
                }
                depths.push(raises);
            }
            depths
        };

        let config = BetConfig {
            sizes: [
                vec![], // preflop (unused in postflop solve)
                build_street(per_street.flop_bet, per_street.flop_raise),
                build_street(per_street.turn_bet, per_street.turn_raise),
                build_street(per_street.river_bet, per_street.river_raise),
            ],
            max_raises,
            allin_threshold,
            allin_pot_ratio,
            no_donk,
            geometric_2bets: geometric,
        };
        Some(Arc::new(config))
    } else if allin_pot_ratio > 0.0 || allin_threshold > 0.0 || max_raises != 3 || no_donk || geometric {
        let mut config = BetConfig::default();
        config.allin_pot_ratio = allin_pot_ratio;
        config.allin_threshold = allin_threshold;
        config.max_raises = max_raises;
        config.no_donk = no_donk;
        config.geometric_2bets = geometric;
        Some(Arc::new(config))
    } else {
        None
    };

    println!("Solving {} spot: board={} pot={} stack={}", street_str, board_str, pot, stack);
    println!("  OOP range: {} combos", oop_range.count_live());
    println!("  IP range: {} combos", ip_range.count_live());
    println!("  Iterations: {}", iterations);
    if let Some(ref bc) = bet_config {
        println!("  Custom bet config: {:?}", bc.sizes[street.index()]);
    }

    if let Some(dl) = depth_limit {
        println!("  Depth limit: stop at {:?}", dl);
    }

    let config = SubgameConfig {
        board,
        pot,
        stacks: [stack, stack],
        ranges: [oop_range, ip_range],
        iterations,
        street,
        warmup_frac: 0.0,
        bet_config,
        dcfr: if algorithm == "qre" { false } else { !no_dcfr },
        cfr_plus: if algorithm == "qre" { false } else { true },
        skip_cum_strategy: if algorithm == "qre" { false } else { skip_cum_strategy },
        dcfr_mode: if lcfr { DcfrMode::Linear } else if dcfr_gamma >= 0.0 || dcfr_alpha >= 0.0 {
            let a = if dcfr_alpha >= 0.0 { dcfr_alpha } else { 1.5 };
            let g = if dcfr_gamma >= 0.0 { dcfr_gamma } else { 2.0 };
            DcfrMode::Custom(a, 0.0, g)
        } else { DcfrMode::Standard },
        depth_limit,
        rake_pct,
        rake_cap,
        exploration_eps,
        entropy_bonus,
        entropy_anneal,
        entropy_root_only,
        opp_dilute,
        softmax_temp: if algorithm == "qre" && softmax_temp <= 0.0 { 10.0 } else { softmax_temp },
        current_iteration: 0,
        use_iso: !no_iso,
        rm_floor: rm_floor,
        alternating,
        t_weight,
        frozen_root: None,
        check_bias,
        pref_passive_delta: pref_delta,
        pref_beta,
        pref_beta_all_nodes: pref_beta_all,
        pruning,
        combo_check_bias: None, frozen_warmup, unfreeze_decay,
    };

    if no_iso {
        println!("  ISO: DISABLED (all cards enumerated individually)");
    }

    let mut solver = SubgameSolver::new(config);

    // Load frozen root strategy from GTO+ file
    if let Some(fr_path) = frozen_root_file {
        println!("  Loading frozen root from: {}", fr_path);
        let frozen = parse_gtoplus_frozen_root(fr_path);
        solver.config.frozen_root = Some(frozen);
    }

    // Load per-combo check bias from JSON file
    if let Some(bias_path) = combo_bias_file {
        use dcfr_solver::card::NUM_COMBOS;
        println!("  Loading combo check bias from: {}", bias_path);
        let bias_json = std::fs::read_to_string(bias_path).expect("cannot read combo bias file");
        let bias_vec: Vec<f32> = serde_json::from_str(&bias_json).expect("invalid combo bias JSON (expected array of 1326 floats)");
        assert_eq!(bias_vec.len(), NUM_COMBOS, "combo bias must have exactly 1326 entries");
        let mut bias_arr = Box::new([0.0f32; NUM_COMBOS]);
        bias_arr.copy_from_slice(&bias_vec);
        solver.config.combo_check_bias = Some(bias_arr);
    }

    // Load valuenet if provided (requires nn feature)
    #[cfg(feature = "nn")]
    if let Some(vn_path) = _valuenet_path {
        println!("  Loading valuenet: {}", vn_path);
        let vn = std::sync::Arc::new(dcfr_solver::valuenet::ValueNet::load(vn_path));
        solver.set_valuenet(vn);
    }

    if algorithm == "egt" {
        println!("  Algorithm: EGT (Excessive Gap Technique → MaxEnt NE)");
        solver.egt_solve(|iter, s| {
            let expl = s.exploitability_pct();
            println!("  iter {:>5}: exploitability = {:.4}% pot ({} nodes)", iter, expl, s.num_decision_nodes());
        });
    } else if algorithm == "qre" {
        println!("  Algorithm: QRE (vanilla CFR + softmax, temp={:.1})", solver.config.softmax_temp);
        solver.solve_with_callback(|iter, s| {
            let expl = s.exploitability_pct();
            println!("  iter {:>5}: exploitability = {:.4}% pot ({} nodes)", iter, expl, s.num_decision_nodes());
        });
    } else if algorithm == "qre2" {
        println!("  Algorithm: QRE2 (fixed-point, λ={}, damping={}, anneal={})", qre_lambda, qre_damping, qre_anneal);
        solver.qre_solve(qre_lambda, qre_damping, qre_anneal, |iter, s| {
            let expl = s.exploitability_pct();
            println!("  iter {:>5}: exploitability = {:.4}% pot ({} nodes)", iter, expl, s.num_decision_nodes());
        });
    } else {
        let mut prev_snap: Option<Vec<f32>> = None;
        solver.solve_with_callback(|iter, s| {
            let expl_avg = s.exploitability_pct();
            let expl_cur = s.exploitability_current_pct();
            let (avg_reg, nz_frac, drift) = s.regret_policy_stats(prev_snap.as_deref());
            println!("  iter {:>5}: avg_expl={:.4}% cur_expl={:.4}% reg={:.4} nz={:.2}% drift={:.4} ({} nodes)",
                iter, expl_avg, expl_cur, avg_reg, nz_frac * 100.0, drift, s.num_decision_nodes());
            prev_snap = Some(s.root_strat_snapshot());
        });
    }

    println!("  Final: {} decision nodes", solver.num_decision_nodes());

    // Two-phase solve: use phase 1 results to build frozen root, then re-solve.
    if two_phase > 0.0 && frozen_root_file.is_none() {
        println!("\n  === Two-phase solve (threshold={:.2} chips) ===", two_phase);
        if let Some((bet_frac, ev_gap)) = solver.root_bet_fractions() {
            use dcfr_solver::card::NUM_COMBOS;
            let mut frozen = Box::new([0.5f32; NUM_COMBOS]);
            let mut indiff_count = 0;
            let mut kept_count = 0;
            for i in 0..NUM_COMBOS {
                if ev_gap[i] == 0.0 && bet_frac[i] == 0.0 { continue; } // inactive
                if ev_gap[i].abs() < two_phase {
                    // Indifferent: set to 50/50 (configurable target)
                    frozen[i] = 0.5;
                    indiff_count += 1;
                } else {
                    // Keep solver's strategy
                    frozen[i] = bet_frac[i];
                    kept_count += 1;
                }
            }
            println!("  Phase 1 done: {} indifferent → 50/50, {} kept", indiff_count, kept_count);

            // Phase 2: re-solve with frozen root
            let mut config2 = SubgameConfig {
                board, pot, stacks: [stack, stack],
                ranges: [solver.config.ranges[0].clone(), solver.config.ranges[1].clone()],
                iterations, street, warmup_frac: 0.0, bet_config: solver.config.bet_config.clone(),
                dcfr: solver.config.dcfr, cfr_plus: solver.config.cfr_plus,
                skip_cum_strategy: solver.config.skip_cum_strategy,
                dcfr_mode: solver.config.dcfr_mode.clone(),
                depth_limit: solver.config.depth_limit.clone(),
                rake_pct: solver.config.rake_pct, rake_cap: solver.config.rake_cap,
                exploration_eps: solver.config.exploration_eps,
                entropy_bonus: solver.config.entropy_bonus, entropy_anneal: solver.config.entropy_anneal,
                entropy_root_only: solver.config.entropy_root_only,
                opp_dilute: solver.config.opp_dilute,
                softmax_temp: solver.config.softmax_temp,
                current_iteration: 0, use_iso: solver.config.use_iso,
                rm_floor: solver.config.rm_floor, alternating: solver.config.alternating,
                t_weight: solver.config.t_weight,
                frozen_root: Some(frozen),
                check_bias: 0.0,
                pref_passive_delta: 1.0, pref_beta: 0.0, pref_beta_all_nodes: false, pruning: false, combo_check_bias: None, frozen_warmup: 0, unfreeze_decay: 1.0,
            };
            let mut solver2 = SubgameSolver::new(config2);
            println!("  Phase 2: re-solving with frozen root ({} iterations)...", iterations);
            solver2.solve();
            let expl2 = solver2.exploitability_pct();
            let (ev2_oop, ev2_ip) = solver2.overall_ev();
            println!("  Phase 2 done: expl={:.4}%, OOP_EV={:.2}, IP_EV={:.2}", expl2, ev2_oop, ev2_ip);
            solver = solver2;
        }
    }

    let expl_pct = solver.exploitability_pct();

    // Post-processing: smooth indifferent hands (root + first child nodes only)
    if smooth_threshold > 0.0 {
        println!("  Smoothing strategies (threshold={:.2} chips, depth≤1)...", smooth_threshold);
        let n = solver.smooth_strategies_depth(smooth_threshold, 1);
        println!("  Smoothed {} total combos", n);
        let expl_after = solver.exploitability_pct();
        println!("  Exploitability after smoothing: {:.4}% pot (was {:.4}%)", expl_after, expl_pct);
    }

    // Post-processing: passive tie-break (before purify, since purify clips low freqs)
    if passive_tiebreak > 0.0 {
        println!("  Passive tie-break (EV threshold={:.2} chips, blend=1.0)...", passive_tiebreak);
        let n = solver.passive_tiebreak(passive_tiebreak, 1.0);
        println!("  Adjusted {} combos toward passive action", n);
        let expl_after = solver.exploitability_pct();
        println!("  Exploitability after tie-break: {:.4}% pot", expl_after);
    }

    // Post-processing: purify low-frequency actions
    if purify_pct > 0.0 {
        println!("  Purifying actions below {:.1}%...", purify_pct);
        let n = solver.purify(purify_pct);
        println!("  Purified {} combos", n);
        let expl_after = solver.exploitability_pct();
        println!("  Exploitability after purification: {:.4}% pot", expl_after);
    }

    let (oop_ev, ip_ev) = solver.overall_ev();
    println!("  OOP EV: {:.2} | IP EV: {:.2} | Sum: {:.2} (pot: {})",
        oop_ev, ip_ev, oop_ev + ip_ev, solver.config.pot);

    solver.root_ev_analysis();

    let mut result = SolveResult::from_solver(&solver);
    // Re-compute exploitability AFTER post-processing (smooth/tiebreak/purify may change it)
    result.exploitability_pct = Some(solver.exploitability_pct());

    // Determine output format from --format flag or file extension
    let use_html = match format_str {
        Some("html") => true,
        Some("json") => false,
        _ => output.ends_with(".html") || output.ends_with(".htm"),
    };

    if use_html {
        let html = result.to_html();
        fs::write(output, &html).expect("cannot write output");
        println!("Saved HTML to {}", output);
    } else {
        let json = result.to_json();
        fs::write(output, &json).expect("cannot write output");
        println!("Saved JSON to {}", output);
    }
}

/// Parse a comma-separated list of pot-fraction percentages into BetSize values.
/// E.g., "33,75,150" → [Frac(1,3), Frac(3,4), Frac(3,2)]
fn parse_size_list(s: Option<&str>) -> Vec<BetSize> {
    let s = match s {
        Some(s) if !s.is_empty() => s,
        _ => return vec![],
    };
    s.split(',')
        .filter_map(|tok| {
            let pct: i32 = tok.trim().parse().ok()?;
            if pct <= 0 { return None; }
            // Simplify common fractions
            let (n, d) = simplify_frac(pct, 100);
            Some(BetSize::Frac(n, d))
        })
        .collect()
}

/// Simplify n/d by GCD.
fn simplify_frac(n: i32, d: i32) -> (i32, i32) {
    let g = gcd(n.unsigned_abs(), d.unsigned_abs()) as i32;
    (n / g, d / g)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn cmd_preflop(
    iterations: u64,
    output: &str,
    chart_output: Option<&str>,
    matchup_output: Option<&str>,
    seed: u64,
    open_size: i32,
    sb_open_size: i32,
    bet3_size: i32,
    bet4_size: i32,
    sb_limp: bool,
    oop_pot_tax: f32,
) {
    let config = PreflopBetConfig {
        raise_sizes: vec![
            vec![open_size],   // depth 0 (open)
            vec![bet3_size],   // depth 1 (3-bet)
            vec![bet4_size],   // depth 2 (4-bet)
        ],
        sb_limp,
        sb_open_size: Some(sb_open_size),
        min_allin_depth: 1,  // No open-shove; all-in allowed after first raise
    };

    println!("Training 6-max preflop MCCFR:");
    println!("  iterations: {}", iterations);
    println!("  open size: {} chips ({}bb)", open_size, open_size as f32 / 2.0);
    println!("  SB open: {} chips ({}bb)", sb_open_size, sb_open_size as f32 / 2.0);
    println!("  3-bet: {} chips ({}bb)", bet3_size, bet3_size as f32 / 2.0);
    println!("  4-bet: {} chips ({}bb)", bet4_size, bet4_size as f32 / 2.0);
    println!("  SB limp: {}", sb_limp);
    println!("  OOP pot tax: {:.0}%", oop_pot_tax * 100.0);
    println!("  seed: {}", seed);

    let mut trainer = PreflopTrainer::new(config, seed);
    trainer.oop_pot_tax = oop_pot_tax;
    let start = std::time::Instant::now();
    trainer.train(iterations);
    let elapsed = start.elapsed();

    println!("Done in {:.1}s. {} info sets.",
        elapsed.as_secs_f64(), trainer.blueprint.entries.len());
    println!("  {:.0} iterations/sec",
        iterations as f64 / elapsed.as_secs_f64());

    println!("Saving blueprint to {}...", output);
    let file = fs::File::create(output).expect("cannot create output file");
    let mut writer = BufWriter::new(file);
    trainer.blueprint.save(&mut writer).expect("failed to save blueprint");

    if let Some(chart_path) = chart_output {
        println!("Extracting charts to {}...", chart_path);
        let charts = trainer.blueprint.extract_charts();
        let json = serde_json::to_string_pretty(&charts).expect("failed to serialize charts");
        fs::write(chart_path, &json).expect("cannot write chart output");
    }

    if let Some(matchup_path) = matchup_output {
        println!("Extracting matchups (SRP + 3bet + 4bet) to {}...", matchup_path);
        let matchups = trainer.blueprint.extract_all_matchups();
        let json = serde_json::to_string_pretty(&matchups).expect("failed to serialize matchups");
        fs::write(matchup_path, &json).expect("cannot write matchup output");
        println!("Saved {} matchups.", matchups.len());
    }

    println!("Done.");
}

fn cmd_chart(blueprint_path: &str, output: &str) {
    println!("Loading blueprint from {}...", blueprint_path);
    let file = fs::File::open(blueprint_path).expect("cannot open blueprint file");
    let mut reader = BufReader::new(file);
    let blueprint = dcfr_solver::mccfr::Blueprint::load(&mut reader)
        .expect("failed to load blueprint");

    println!("Extracting preflop chart ({} info sets)...", blueprint.entries.len());
    let chart = PreflopChart::from_blueprint(&blueprint);

    let json = export_preflop_chart(&chart);
    fs::write(output, &json).expect("cannot write output");
    println!("Saved to {}", output);
}

fn cmd_chart_preflop(blueprint_path: &str, output: &str, matchup_output: Option<&str>) {
    use dcfr_solver::preflop::PreflopBlueprint;

    println!("Loading preflop blueprint from {}...", blueprint_path);
    let file = fs::File::open(blueprint_path).expect("cannot open blueprint file");
    let mut reader = BufReader::new(file);
    let blueprint = PreflopBlueprint::load(&mut reader)
        .expect("failed to load preflop blueprint");

    println!("  {} iterations, {} info sets", blueprint.iterations, blueprint.entries.len());
    println!("  config: {:?}", blueprint.config);

    println!("Extracting RFI charts...");
    let charts = blueprint.extract_charts();

    let json = serde_json::to_string_pretty(&charts).expect("failed to serialize charts");
    fs::write(output, &json).expect("cannot write output");
    println!("Saved {} spots to {}", charts.len(), output);

    if let Some(matchup_path) = matchup_output {
        println!("Extracting matchups (SRP + 3bet + 4bet) to {}...", matchup_path);
        let matchups = blueprint.extract_all_matchups();
        let json = serde_json::to_string_pretty(&matchups).expect("failed to serialize matchups");
        fs::write(matchup_path, &json).expect("cannot write matchup output");
        println!("Saved {} matchups to {}", matchups.len(), matchup_path);
    }
}

fn cmd_batch_config(
    matchups_path: &str,
    generate_template: Option<&str>,
    defender_ranges_path: Option<&str>,
    iterations: u32,
    output: &str,
) {
    use dcfr_solver::preflop::SrpMatchup;

    println!("Loading matchups from {}...", matchups_path);
    let content = fs::read_to_string(matchups_path).expect("cannot read matchups file");
    let matchups: Vec<SrpMatchup> = serde_json::from_str(&content)
        .expect("invalid matchups JSON");
    println!("  {} matchups loaded", matchups.len());

    // Generate template mode
    if let Some(template_path) = generate_template {
        let template = batch::generate_defender_template(&matchups);
        let json = serde_json::to_string_pretty(&template)
            .expect("failed to serialize template");
        fs::write(template_path, &json).expect("cannot write template");
        println!("Defender range template saved to {}", template_path);
        println!("Edit this file and fill in defender ranges, then re-run with --defender-ranges.");
        return;
    }

    // Load defender overrides (if any)
    let defender_ranges = if let Some(path) = defender_ranges_path {
        println!("Loading defender ranges from {}...", path);
        let ranges = batch::load_defender_ranges(path)
            .unwrap_or_else(|e| panic!("defender range error: {}", e));
        let override_count = ranges.values().filter(|v| !v.is_empty()).count();
        println!("  {} overrides loaded", override_count);
        ranges
    } else {
        HashMap::new()
    };

    // Generate batch configs
    println!("Generating batch configs ({} matchups × 1,755 flops)...", matchups.len());
    let entries = batch::generate_batch(&matchups, &defender_ranges, iterations);
    println!("  {} total configs", entries.len());

    // Write JSONL
    let file = fs::File::create(output).expect("cannot create output file");
    let mut writer = std::io::BufWriter::new(file);
    for entry in &entries {
        let line = serde_json::to_string(entry).expect("failed to serialize entry");
        writeln!(writer, "{}", line).expect("write failed");
    }
    writer.flush().expect("flush failed");
    println!("Saved to {}", output);
}

fn cmd_batch_run(
    input: &str,
    output_dir: &str,
    bet_sizes_str: Option<&str>,
    raise_sizes_str: Option<&str>,
    start: usize,
    count: usize,
    skip_existing: bool,
    skip_cum_strategy: bool,
) {
    use dcfr_solver::batch::BatchEntry;

    // Parse bet config (shared across all spots).
    // Default: Config D (bet 33%+67%+125%, raise 50%+100%, allin≤3×pot).
    let bet_config = if bet_sizes_str.is_some() || raise_sizes_str.is_some() {
        let bet_sizes = parse_size_list(bet_sizes_str);
        let raise_sizes = parse_size_list(raise_sizes_str);
        let mut config = BetConfig::default();
        for street_idx in 1..=3 {
            if !bet_sizes.is_empty() {
                if config.sizes[street_idx].is_empty() {
                    config.sizes[street_idx].push(bet_sizes.clone());
                } else {
                    config.sizes[street_idx][0] = bet_sizes.clone();
                }
            }
            if !raise_sizes.is_empty() {
                if config.sizes[street_idx].len() < 2 {
                    config.sizes[street_idx].resize(2, vec![]);
                    config.sizes[street_idx][1] = raise_sizes.clone();
                } else {
                    config.sizes[street_idx][1] = raise_sizes.clone();
                }
            }
        }
        Some(Arc::new(config))
    } else {
        // Config D: bet 33%+67%+125%, raise 50%+100%, max_raises=2, allin≤3×pot
        let street_sizes = vec![
            vec![BetSize::Frac(33, 100), BetSize::Frac(2, 3), BetSize::Frac(5, 4)],
            vec![BetSize::Frac(1, 2), BetSize::Frac(1, 1)],
        ];
        Some(Arc::new(BetConfig {
            sizes: [vec![], street_sizes.clone(), street_sizes.clone(), street_sizes],
            max_raises: 2,
            allin_threshold: 0.67,
            allin_pot_ratio: 3.0,
            no_donk: false,
            geometric_2bets: false,
        }))
    };

    // Read JSONL entries
    println!("Loading batch configs from {}...", input);
    let file = fs::File::open(input).expect("cannot open input file");
    let reader = BufReader::new(file);
    let all_entries: Vec<BatchEntry> = reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let line = line.trim();
            if line.is_empty() { return None; }
            serde_json::from_str(line).ok()
        })
        .collect();
    println!("  {} total configs loaded", all_entries.len());

    // Slice based on start/count
    let start = start.min(all_entries.len());
    let end = if count == 0 { all_entries.len() } else { (start + count).min(all_entries.len()) };
    let entries = &all_entries[start..end];
    println!("  Processing entries {}..{} ({} spots)", start, end, entries.len());
    if let Some(ref bc) = bet_config {
        println!("  Bet config: {:?}", bc.sizes[1]);
        println!("  max_raises={}, allin_threshold={}, allin_pot_ratio={}",
            bc.max_raises, bc.allin_threshold, bc.allin_pot_ratio);
    }
    println!("  skip_cum_strategy={}", skip_cum_strategy);

    // Create output directory
    fs::create_dir_all(output_dir).expect("cannot create output directory");

    let total = entries.len();
    let mut solved = 0usize;
    let mut skipped = 0usize;
    let batch_start = Instant::now();

    for (i, entry) in entries.iter().enumerate() {
        let global_idx = start + i;

        // Output filename: {matchup}_{board}_{idx}.json
        let safe_matchup = entry.matchup.replace(' ', "_");
        let filename = format!("{}_{}.json", safe_matchup, entry.board);
        let output_path = Path::new(output_dir).join(&filename);

        // Skip existing
        if skip_existing && output_path.exists() {
            skipped += 1;
            continue;
        }

        // Parse board
        let board_cards = match parse_cards(&entry.board) {
            Some(cards) => cards,
            None => {
                eprintln!("  [{}] SKIP: invalid board '{}'", global_idx, entry.board);
                continue;
            }
        };
        let mut board = Hand::new();
        for c in &board_cards {
            board = board.add(*c);
        }

        let street = match entry.street.to_lowercase().as_str() {
            "flop" => Street::Flop,
            "turn" => Street::Turn,
            "river" => Street::River,
            _ => {
                eprintln!("  [{}] SKIP: invalid street '{}'", global_idx, entry.street);
                continue;
            }
        };

        // Parse ranges
        let oop_range = match Range::parse(&entry.oop_range) {
            Some(r) => r,
            None => {
                eprintln!("  [{}] SKIP: invalid OOP range for {}", global_idx, entry.matchup);
                continue;
            }
        };
        let ip_range = match Range::parse(&entry.ip_range) {
            Some(r) => r,
            None => {
                eprintln!("  [{}] SKIP: invalid IP range for {}", global_idx, entry.matchup);
                continue;
            }
        };

        let config = SubgameConfig {
            board,
            pot: entry.pot,
            stacks: [entry.stack, entry.stack],
            ranges: [oop_range, ip_range],
            iterations: entry.iterations,
            street,
            warmup_frac: 0.0,
            bet_config: bet_config.clone(),
            dcfr: true,
            cfr_plus: true,
            skip_cum_strategy,
            dcfr_mode: DcfrMode::Standard,
            depth_limit: None,
            rake_pct: 0.0,
            rake_cap: 0.0,
            exploration_eps: 0.0,
            entropy_bonus: 0.0,
            entropy_anneal: false, entropy_root_only: false,
            opp_dilute: 0.0,
            softmax_temp: 0.0,
            current_iteration: 0,
            use_iso: true,
            rm_floor: 0.0, alternating: false, t_weight: false, frozen_root: None, check_bias: 0.0, pref_passive_delta: 1.0, pref_beta: 0.0, pref_beta_all_nodes: false, pruning: false, combo_check_bias: None, frozen_warmup: 0, unfreeze_decay: 1.0,
        };

        let spot_start = Instant::now();
        let mut solver = SubgameSolver::new(config);
        solver.solve();

        let expl = solver.exploitability_pct();
        let mut result = SolveResult::from_solver(&solver);
        result.exploitability_pct = Some(expl);
        let json = result.to_json();
        fs::write(&output_path, &json).expect("cannot write result");

        solved += 1;
        let spot_secs = spot_start.elapsed().as_secs_f64();
        let total_secs = batch_start.elapsed().as_secs_f64();
        let remaining = entries.len() - (i + 1);
        let avg_secs = if solved > 0 { total_secs / solved as f64 } else { spot_secs };
        let eta_secs = avg_secs * remaining as f64;

        eprintln!(
            "  [{}/{}] {} {} | {:.1}s | expl={:.3}% | ETA {:.0}m | saved {}",
            i + 1, total, entry.matchup, entry.board,
            spot_secs, expl, eta_secs / 60.0, filename,
        );
    }

    let total_secs = batch_start.elapsed().as_secs_f64();
    println!("\n=== Batch complete ===");
    println!("  Solved: {}", solved);
    println!("  Skipped: {}", skipped);
    println!("  Total time: {:.1}s ({:.1}m)", total_secs, total_secs / 60.0);
    if solved > 0 {
        println!("  Avg per spot: {:.1}s", total_secs / solved as f64);
    }
}

#[cfg(feature = "nn")]
fn cmd_datagen(
    street_str: &str,
    count: usize,
    iterations: u32,
    output_dir: &str,
    matchups_path: Option<&str>,
    seed: u64,
    min_spr: f32,
    max_spr: f32,
) {
    use dcfr_solver::datagen::{DatagenConfig, run_datagen};

    let street = match street_str.to_lowercase().as_str() {
        "turn" => Street::Turn,
        "river" => Street::River,
        _ => panic!("datagen only supports turn/river, got: {}", street_str),
    };

    let config = DatagenConfig {
        count,
        street,
        iterations,
        output_dir: output_dir.to_string(),
        seed,
        matchups_path: matchups_path.map(|s| s.to_string()),
        min_spr,
        max_spr,
    };

    run_datagen(config);
}
