/// Equity abstraction — bucket hands by equity for Blueprint engine.
///
/// Precomputes equity for every canonical (hole, board) combination
/// on flop and turn, then maps to 0..49 buckets via percentile binning.
/// River equity is lazily cached during MCCFR training.

use crate::card::{Card, Hand, NUM_CARDS};
use crate::eval::evaluate;
use crate::iso::canonical_observation;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::sync::RwLock;

pub const N_BUCKETS: u8 = 50;
pub const ABSTRACTION_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// EquityAbstraction
// ---------------------------------------------------------------------------

pub struct EquityAbstraction {
    pub flop: HashMap<i64, u8>,
    pub turn: HashMap<i64, u8>,
    pub river_cache: RwLock<HashMap<i64, u8>>,
}

impl EquityAbstraction {
    pub fn new() -> Self {
        EquityAbstraction {
            flop: HashMap::new(),
            turn: HashMap::new(),
            river_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get bucket for a hand on a given street.
    pub fn bucket(&self, hole: Hand, board: Hand, street: crate::game::Street) -> u8 {
        let key = canonical_observation(hole, board);
        match street {
            crate::game::Street::Preflop => {
                // Preflop uses canonical hand index (0..168), handled by infoset
                unreachable!("preflop doesn't use equity abstraction")
            }
            crate::game::Street::Flop => {
                *self.flop.get(&key).unwrap_or(&0)
            }
            crate::game::Street::Turn => {
                *self.turn.get(&key).unwrap_or(&0)
            }
            crate::game::Street::River => {
                // Check cache first
                {
                    let cache = self.river_cache.read().unwrap();
                    if let Some(&b) = cache.get(&key) {
                        return b;
                    }
                }
                // Compute and cache
                let eq = river_equity(hole, board);
                let b = (eq * (N_BUCKETS as f32 - 0.001)).min(N_BUCKETS as f32 - 1.0) as u8;
                {
                    let mut cache = self.river_cache.write().unwrap();
                    cache.insert(key, b);
                }
                b
            }
        }
    }

    /// Build abstraction by computing equity tables for flop and turn.
    pub fn build() -> Self {
        println!("Building equity abstraction...");

        println!("  Computing flop equities...");
        let flop = build_street_table(3);

        println!("  Computing turn equities...");
        let turn = build_street_table(4);

        println!("  Abstraction built: {} flop entries, {} turn entries",
                 flop.len(), turn.len());

        EquityAbstraction {
            flop,
            turn,
            river_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Save to binary file.
    pub fn save<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(ABSTRACTION_VERSION)?;

        // Flop
        w.write_u64::<LittleEndian>(self.flop.len() as u64)?;
        for (&key, &bucket) in &self.flop {
            w.write_i64::<LittleEndian>(key)?;
            w.write_u8(bucket)?;
        }

        // Turn
        w.write_u64::<LittleEndian>(self.turn.len() as u64)?;
        for (&key, &bucket) in &self.turn {
            w.write_i64::<LittleEndian>(key)?;
            w.write_u8(bucket)?;
        }

        Ok(())
    }

    /// Load from binary file.
    pub fn load<R: Read>(r: &mut R) -> io::Result<Self> {
        let version = r.read_u32::<LittleEndian>()?;
        if version != ABSTRACTION_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                      format!("version mismatch: expected {}, got {}", ABSTRACTION_VERSION, version)));
        }

        // Flop
        let n = r.read_u64::<LittleEndian>()? as usize;
        let mut flop = HashMap::with_capacity(n);
        for _ in 0..n {
            let key = r.read_i64::<LittleEndian>()?;
            let bucket = r.read_u8()?;
            flop.insert(key, bucket);
        }

        // Turn
        let n = r.read_u64::<LittleEndian>()? as usize;
        let mut turn = HashMap::with_capacity(n);
        for _ in 0..n {
            let key = r.read_i64::<LittleEndian>()?;
            let bucket = r.read_u8()?;
            turn.insert(key, bucket);
        }

        Ok(EquityAbstraction {
            flop,
            turn,
            river_cache: RwLock::new(HashMap::new()),
        })
    }
}

// ---------------------------------------------------------------------------
// Equity computation
// ---------------------------------------------------------------------------

/// River equity: enumerate all C(45,2)=990 opponent hands.
pub fn river_equity(hole: Hand, board: Hand) -> f32 {
    let my_hand = hole.union(board);
    let my_strength = evaluate(my_hand);
    let dead = hole.union(board);

    let mut wins = 0.0f64;
    let mut total = 0u32;

    for o1 in 0..NUM_CARDS as Card {
        if dead.contains(o1) { continue; }
        for o2 in (o1 + 1)..NUM_CARDS as Card {
            if dead.contains(o2) { continue; }
            let opp_hand = board.union(Hand::from_card(o1).add(o2));
            let opp_strength = evaluate(opp_hand);

            if my_strength > opp_strength {
                wins += 1.0;
            } else if my_strength == opp_strength {
                wins += 0.5;
            }
            total += 1;
        }
    }

    (wins / total as f64) as f32
}

/// Turn equity: enumerate 46 river cards × C(45,2) opponents.
pub fn turn_equity(hole: Hand, board: Hand) -> f32 {
    let dead = hole.union(board);
    let mut total_eq = 0.0f64;
    let mut n_rivers = 0u32;

    for river_card in 0..NUM_CARDS as Card {
        if dead.contains(river_card) { continue; }
        let river_board = board.add(river_card);
        let eq = river_equity(hole, river_board);
        total_eq += eq as f64;
        n_rivers += 1;
    }

    (total_eq / n_rivers as f64) as f32
}

/// Flop equity: enumerate C(47,2)=1081 turn+river combos × opponents.
/// This is expensive! ~1M evaluations per hand.
pub fn flop_equity(hole: Hand, board: Hand) -> f32 {
    let dead = hole.union(board);
    let mut total_eq = 0.0f64;
    let mut n_runouts = 0u32;

    for turn_card in 0..NUM_CARDS as Card {
        if dead.contains(turn_card) { continue; }
        let turn_board = board.add(turn_card);
        let turn_dead = dead.add(turn_card);

        for river_card in (turn_card + 1)..NUM_CARDS as Card {
            if turn_dead.contains(river_card) { continue; }
            let river_board = turn_board.add(river_card);
            let eq = river_equity(hole, river_board);
            total_eq += eq as f64;
            n_runouts += 1;
        }
    }

    (total_eq / n_runouts as f64) as f32
}

// ---------------------------------------------------------------------------
// Table building
// ---------------------------------------------------------------------------

fn build_street_table(board_size: usize) -> HashMap<i64, u8> {
    // Enumerate all canonical (hole, board) combinations

    // Generate all boards of given size
    let boards = enumerate_boards(board_size);
    let n_boards = boards.len();
    println!("    {} boards to process", n_boards);

    // Process boards in parallel
    let all_entries: Vec<Vec<(i64, f32)>> = boards.par_iter().map(|board| {
        let mut local = Vec::new();
        let dead = *board;

        for c1 in 0..NUM_CARDS as Card {
            if dead.contains(c1) { continue; }
            for c2 in (c1 + 1)..NUM_CARDS as Card {
                if dead.contains(c2) { continue; }
                let hole = Hand::from_card(c1).add(c2);
                let key = canonical_observation(hole, *board);

                let eq = if board_size == 3 {
                    flop_equity(hole, *board)
                } else {
                    turn_equity(hole, *board)
                };

                local.push((key, eq));
            }
        }
        local
    }).collect();

    // Merge and deduplicate (canonical keys collapse duplicates)
    let mut key_equity: HashMap<i64, f32> = HashMap::new();
    for batch in all_entries {
        for (key, eq) in batch {
            key_equity.entry(key).or_insert(eq);
        }
    }

    // Sort by equity and assign buckets via percentile binning
    let mut sorted: Vec<(i64, f32)> = key_equity.into_iter().collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n = sorted.len();
    let mut result = HashMap::with_capacity(n);
    for (i, &(key, _eq)) in sorted.iter().enumerate() {
        let bucket = ((i as f64 / n as f64) * N_BUCKETS as f64).min(N_BUCKETS as f64 - 1.0) as u8;
        result.insert(key, bucket);
    }

    result
}

fn enumerate_boards(size: usize) -> Vec<Hand> {
    let mut boards = Vec::new();
    match size {
        3 => {
            for c1 in 0..52u8 {
                for c2 in (c1 + 1)..52u8 {
                    for c3 in (c2 + 1)..52u8 {
                        boards.push(Hand::from_card(c1).add(c2).add(c3));
                    }
                }
            }
        }
        4 => {
            for c1 in 0..52u8 {
                for c2 in (c1 + 1)..52u8 {
                    for c3 in (c2 + 1)..52u8 {
                        for c4 in (c3 + 1)..52u8 {
                            boards.push(Hand::from_card(c1).add(c2).add(c3).add(c4));
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    boards
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card;

    #[test]
    fn test_river_equity_nuts() {
        // AA on board AsKhTd5c2s — top set
        let hole = Hand::from_card(card(12, 0)).add(card(12, 1)); // Ac Ad
        let board = Hand::from_card(card(12, 3)) // As
            .add(card(11, 2)) // Kh
            .add(card(8, 1))  // Td
            .add(card(3, 0))  // 5c
            .add(card(0, 3)); // 2s
        let eq = river_equity(hole, board);
        // AA should have very high equity on this board
        assert!(eq > 0.9, "AA equity should be > 0.9, got {}", eq);
    }

    #[test]
    fn test_river_equity_air() {
        // 33 on board AsKhTd5c2s — underpair
        let hole = Hand::from_card(card(1, 0)).add(card(1, 1)); // 3c 3d
        let board = Hand::from_card(card(12, 3))
            .add(card(11, 2))
            .add(card(8, 1))
            .add(card(3, 0))
            .add(card(0, 3));
        let eq = river_equity(hole, board);
        // 33 should have low equity
        assert!(eq < 0.5, "33 equity should be < 0.5, got {}", eq);
    }

    #[test]
    fn test_bucket_range() {
        let eq = 0.75f32;
        let bucket = (eq * (N_BUCKETS as f32 - 0.001)).min(N_BUCKETS as f32 - 1.0) as u8;
        assert!(bucket < N_BUCKETS);
        assert_eq!(bucket, 37);
    }
}
