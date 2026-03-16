/// Phase B: ISO ON vs OFF differential test.
/// Compares exploitability and root EV between ISO ON and ISO OFF solvers
/// on paired/monotone/normal boards. If ISO implementation is correct,
/// both should converge to similar exploitability at the same iteration count
/// (adjusted for the larger tree in ISO OFF).
///
/// This test catches bugs in chance_utility aggregation, compose_perms,
/// and remap_iso_perms by comparing final values rather than intermediate state.

use dcfr_solver::cfr::{SubgameSolver, SubgameConfig, DcfrMode};
use dcfr_solver::card::Hand;
use dcfr_solver::game::Street;
use dcfr_solver::range::Range;

fn card(rank: u8, suit: u8) -> u8 {
    rank * 4 + suit
}

fn make_config(board: Hand, street: Street, iterations: u32, use_iso: bool) -> SubgameConfig {
    let range_str = "AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J2s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o";
    let oop_range = Range::parse(range_str).unwrap();
    let ip_range = Range::parse(range_str).unwrap();

    SubgameConfig {
        board,
        pot: 30,
        stacks: [170, 170],
        ranges: [oop_range, ip_range],
        iterations,
        street,
        warmup_frac: 0.0,
        bet_config: None,
        dcfr: true,
        cfr_plus: true,
        skip_cum_strategy: false,
        dcfr_mode: DcfrMode::Standard,
        depth_limit: None,
        rake_pct: 0.0,
        rake_cap: 0.0,
        exploration_eps: 0.0,
        entropy_bonus: 0.0, entropy_anneal: false, entropy_root_only: false,
        opp_dilute: 0.0,
        softmax_temp: 0.0,
        current_iteration: 0,
        use_iso,
        rm_floor: 0.0,
        alternating: false, t_weight: false,
        frozen_root: None, check_bias: 0.0,
        pref_passive_delta: 1.0, pref_beta: 0.0, pref_beta_all_nodes: false, pruning: false,
        combo_check_bias: None, frozen_warmup: 0, unfreeze_decay: 0.0,
    }
}

/// Run ISO ON vs OFF for a board, compare exploitability.
fn compare_iso(board: Hand, name: &str, iterations: u32) {
    println!("\n=== {} (ISO ON vs OFF, {} iter) ===", name, iterations);

    let config_on = make_config(board, Street::Flop, iterations, true);
    let mut solver_on = SubgameSolver::new(config_on);
    solver_on.solve();
    let expl_on = solver_on.exploitability_pct();
    let nodes_on = solver_on.num_decision_nodes();
    let ev_on = solver_on.overall_ev();

    let config_off = make_config(board, Street::Flop, iterations, false);
    let mut solver_off = SubgameSolver::new(config_off);
    solver_off.solve();
    let expl_off = solver_off.exploitability_pct();
    let nodes_off = solver_off.num_decision_nodes();
    let ev_off = solver_off.overall_ev();

    println!("  ISO ON:  {} nodes, expl={:.4}%, OOP_EV={:.4}, IP_EV={:.4}", nodes_on, expl_on, ev_on.0, ev_on.1);
    println!("  ISO OFF: {} nodes, expl={:.4}%, OOP_EV={:.4}, IP_EV={:.4}", nodes_off, expl_off, ev_off.0, ev_off.1);
    println!("  Δexpl = {:.4}%p, ΔEV_OOP = {:.4}, ΔEV_IP = {:.4}",
        expl_on - expl_off, ev_on.0 - ev_off.0, ev_on.1 - ev_off.1);

    // For normal boards, ISO ON should be close to ISO OFF
    // For buggy boards, ISO ON will be much worse
}

#[test]
#[ignore] // Long-running: run with --ignored
fn test_iso_differential_paired() {
    let board = Hand::new()
        .add(card(7, 2))  // 9h
        .add(card(7, 0))  // 9c
        .add(card(3, 1)); // 5d
    compare_iso(board, "9h9c5d (paired)", 300);
}

#[test]
#[ignore]
fn test_iso_differential_monotone() {
    let board = Hand::new()
        .add(card(8, 3))  // Ts
        .add(card(6, 3))  // 8s
        .add(card(1, 3)); // 3s
    compare_iso(board, "Ts8s3s (monotone)", 300);
}

#[test]
#[ignore]
fn test_iso_differential_dynamic() {
    let board = Hand::new()
        .add(card(8, 1))  // Td
        .add(card(7, 1))  // 9d
        .add(card(4, 2)); // 6h
    compare_iso(board, "Td9d6h (dynamic, control)", 300);
}

#[test]
#[ignore]
fn test_iso_differential_twotone() {
    let board = Hand::new()
        .add(card(12, 3)) // As
        .add(card(11, 3)) // Ks
        .add(card(1, 1));  // 3d
    compare_iso(board, "AsKs3d (two-tone, control)", 300);
}

#[test]
#[ignore]
fn test_iso_differential_low_paired() {
    let board = Hand::new()
        .add(card(3, 2))   // 5h
        .add(card(3, 0))   // 5c
        .add(card(11, 1)); // Kd
    compare_iso(board, "5h5cKd (low paired)", 300);
}

#[test]
#[ignore]
fn test_iso_differential_monotone2() {
    let board = Hand::new()
        .add(card(12, 3)) // As
        .add(card(5, 3))  // 7s
        .add(card(0, 3)); // 2s
    compare_iso(board, "As7s2s (monotone2)", 300);
}

#[test]
#[ignore]
fn test_iso_differential_rainbow() {
    let board = Hand::new()
        .add(card(5, 2))  // 7h
        .add(card(3, 3))  // 5s
        .add(card(0, 0)); // 2c
    compare_iso(board, "7h5s2c (rainbow/dry, control)", 300);
}

/// Fast test: 1 iteration on turn boards to isolate chance_utility aggregation.
/// At iteration 1, strategy is uniform (zero regrets), so any EV difference
/// between ISO ON and OFF must come from chance_utility, not regret scatter.
#[test]
fn test_iso_chance_utility_1iter() {
    let boards: Vec<(Hand, &str, Street)> = vec![
        // Turn boards (river-only chance node with ISO)
        (Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)).add(card(11, 3)),
         "9h9c5dKs (paired, turn)", Street::Turn),
        (Hand::new().add(card(8, 3)).add(card(6, 3)).add(card(1, 3)).add(card(10, 0)),
         "Ts8s3sQc (monotone+, turn)", Street::Turn),
        (Hand::new().add(card(8, 1)).add(card(7, 1)).add(card(4, 2)).add(card(11, 3)),
         "Td9d6hKs (normal, turn)", Street::Turn),
        // Flop boards (turn+river chance nodes)
        (Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)),
         "9h9c5d (paired, flop)", Street::Flop),
        (Hand::new().add(card(8, 3)).add(card(6, 3)).add(card(1, 3)),
         "Ts8s3s (monotone, flop)", Street::Flop),
        (Hand::new().add(card(8, 1)).add(card(7, 1)).add(card(4, 2)),
         "Td9d6h (normal, flop)", Street::Flop),
    ];

    println!("\n{}", "=".repeat(80));
    println!("ISO CHANCE_UTILITY ISOLATION TEST — 1 iteration (uniform strategy)");
    println!("{}", "=".repeat(80));
    println!("{:<30} {:>12} {:>12} {:>12} {:>12}",
        "Board", "EV_ON(OOP)", "EV_OFF(OOP)", "ΔOOP", "ΔIP");
    println!("{}", "-".repeat(80));

    let mut any_mismatch = false;
    for (board, name, street) in &boards {
        let config_on = make_config(*board, *street, 1, true);
        let mut solver_on = SubgameSolver::new(config_on);
        solver_on.solve();
        let ev_on = solver_on.overall_ev();

        let config_off = make_config(*board, *street, 1, false);
        let mut solver_off = SubgameSolver::new(config_off);
        solver_off.solve();
        let ev_off = solver_off.overall_ev();

        let d_oop = ev_on.0 - ev_off.0;
        let d_ip = ev_on.1 - ev_off.1;
        let flag = if d_oop.abs() > 0.01 || d_ip.abs() > 0.01 { " *** BUG ***" } else { "" };
        if d_oop.abs() > 0.01 { any_mismatch = true; }

        println!("{:<30} {:>12.4} {:>12.4} {:>12.4} {:>12.4}{}",
            name, ev_on.0, ev_off.0, d_oop, d_ip, flag);
    }
    println!("{}", "=".repeat(80));
    if any_mismatch {
        println!("*** MISMATCH DETECTED: chance_utility aggregation bug confirmed ***");
    } else {
        println!("All EVs match — bug is likely in regret scatter, not chance_utility");
    }
    // Don't assert — this is diagnostic
}

/// Run all boards at once and produce a summary table.
#[test]
#[ignore]
fn test_iso_differential_all_boards() {
    let boards: Vec<(Hand, &str)> = vec![
        // Controls (expect ISO ON ≈ OFF)
        (Hand::new().add(card(8, 1)).add(card(7, 1)).add(card(4, 2)), "Td9d6h dynamic"),
        (Hand::new().add(card(12, 3)).add(card(11, 3)).add(card(1, 1)), "AsKs3d two-tone"),
        (Hand::new().add(card(5, 2)).add(card(3, 3)).add(card(0, 0)), "7h5s2c rainbow"),
        // Problem boards (expect ISO ON >> OFF)
        (Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)), "9h9c5d paired"),
        (Hand::new().add(card(8, 3)).add(card(6, 3)).add(card(1, 3)), "Ts8s3s monotone"),
        // Additional problem candidates
        (Hand::new().add(card(3, 2)).add(card(3, 0)).add(card(11, 1)), "5h5cKd low paired"),
        (Hand::new().add(card(12, 3)).add(card(5, 3)).add(card(0, 3)), "As7s2s monotone2"),
        (Hand::new().add(card(7, 2)).add(card(4, 2)).add(card(2, 2)), "9h6h4h monotone-h"),
    ];

    let iters = 256;
    println!("\n{}", "=".repeat(70));
    println!("ISO DIFFERENTIAL TEST — {} iterations per board", iters);
    println!("{}", "=".repeat(70));
    println!("{:<22} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Board", "Nodes ON", "Nodes OFF", "Expl ON%", "Expl OFF%", "Δexpl%");
    println!("{}", "-".repeat(72));

    for (board, name) in &boards {
        let config_on = make_config(*board, Street::Flop, iters, true);
        let mut solver_on = SubgameSolver::new(config_on);
        solver_on.solve();
        let expl_on = solver_on.exploitability_pct();
        let nodes_on = solver_on.num_decision_nodes();

        let config_off = make_config(*board, Street::Flop, iters, false);
        let mut solver_off = SubgameSolver::new(config_off);
        solver_off.solve();
        let expl_off = solver_off.exploitability_pct();
        let nodes_off = solver_off.num_decision_nodes();

        println!("{:<22} {:>10} {:>10} {:>10.4} {:>10.4} {:>10.4}",
            name, nodes_on, nodes_off, expl_on, expl_off, expl_on - expl_off);
    }
}
