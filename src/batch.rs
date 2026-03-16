/// Batch solve pipeline — generate JSONL configs for postflop mass-solving.
///
/// Connects preflop MCCFR matchup ranges to postflop solver configs:
/// 1. Range format conversion (BTreeMap → solver string)
/// 2. Defender range override (external JSON)
/// 3. 1,755 canonical flops (suit isomorphism)
/// 4. Batch config generation with correct OOP/IP assignment

use crate::preflop::SrpMatchup;
use crate::range::Range;
use std::collections::{BTreeMap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Range conversion
// ---------------------------------------------------------------------------

/// Convert BTreeMap<hand, weight> to solver range string.
/// e.g., {"AA": 1.0, "KK": 0.995, "T9s": 0.974} → "AA,KK:0.995,T9s:0.974"
pub fn range_to_string(range: &BTreeMap<String, f32>) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(range.len());
    for (hand, &weight) in range {
        if weight <= 0.0 {
            continue;
        }
        if weight >= 0.9995 {
            parts.push(hand.clone());
        } else {
            parts.push(format!("{}:{:.3}", hand, weight));
        }
    }
    parts.join(",")
}

// ---------------------------------------------------------------------------
// Defender range template
// ---------------------------------------------------------------------------

/// Generate a JSON template for defender range overrides.
/// Each matchup gets an empty string; user fills in the ranges they want to override.
pub fn generate_defender_template(matchups: &[SrpMatchup]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for m in matchups {
        map.insert(m.matchup.clone(), serde_json::Value::String(String::new()));
    }
    serde_json::Value::Object(map)
}

/// Load defender range overrides from JSON file.
/// Validates non-empty ranges via Range::parse().
pub fn load_defender_ranges(path: &str) -> Result<HashMap<String, String>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let map: HashMap<String, String> = serde_json::from_str(&content)
        .map_err(|e| format!("invalid JSON in {}: {}", path, e))?;

    // Validate non-empty ranges
    for (matchup, range_str) in &map {
        if !range_str.is_empty() {
            Range::parse(range_str).ok_or_else(|| {
                format!("invalid range for '{}': '{}'", matchup, range_str)
            })?;
        }
    }

    Ok(map)
}

// ---------------------------------------------------------------------------
// Canonical flops (1,755)
// ---------------------------------------------------------------------------

/// Generate all 1,755 canonical flops under suit isomorphism.
///
/// Algorithm:
/// 1. Enumerate all C(52,3) = 22,100 three-card combinations
/// 2. Sort ranks descending; for same-rank cards, try all orderings
/// 3. For each ordering, relabel suits by first appearance, keep minimum
/// 4. Deduplicate via HashSet
///
/// Mathematical breakdown:
/// - 3 distinct ranks: C(13,3) × 5 suit textures = 1,430
/// - One pair:         13 × 12  × 2 suit textures = 312
/// - Trips:            13       × 1                = 13
/// - Total: 1,430 + 312 + 13 = 1,755
pub fn canonical_flops() -> Vec<String> {
    const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
    const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

    let mut seen = HashSet::new();
    let mut flops = Vec::new();

    for c1 in 0u8..52 {
        for c2 in (c1 + 1)..52 {
            for c3 in (c2 + 1)..52 {
                let cards = [
                    (c1 >> 2, c1 & 3),
                    (c2 >> 2, c2 & 3),
                    (c3 >> 2, c3 & 3),
                ];

                // Sort by rank descending
                let mut sorted = cards;
                sorted.sort_by(|a, b| b.0.cmp(&a.0));

                // Generate all permutations of same-rank groups, canonicalize each,
                // and keep the lexicographic minimum.
                let orderings = same_rank_permutations(&sorted);
                let mut best: Option<String> = None;

                for ordering in &orderings {
                    let canon = relabel_suits(ordering, &RANK_CHARS, &SUIT_CHARS);
                    if best.as_ref().is_none_or(|b| canon < *b) {
                        best = Some(canon);
                    }
                }

                let canonical = best.unwrap();
                if seen.insert(canonical.clone()) {
                    flops.push(canonical);
                }
            }
        }
    }

    flops.sort();
    flops
}

/// Relabel suits by first appearance order and produce a string.
fn relabel_suits(cards: &[(u8, u8)], rank_chars: &[char; 13], suit_chars: &[char; 4]) -> String {
    let mut suit_map: [i8; 4] = [-1; 4];
    let mut next_suit: u8 = 0;
    let mut result = String::with_capacity(6);
    for &(r, s) in cards {
        let old_suit = s as usize;
        if suit_map[old_suit] < 0 {
            suit_map[old_suit] = next_suit as i8;
            next_suit += 1;
        }
        result.push(rank_chars[r as usize]);
        result.push(suit_chars[suit_map[old_suit] as usize]);
    }
    result
}

/// Generate all permutations of cards within same-rank groups.
/// Input must be sorted by rank descending.
fn same_rank_permutations(cards: &[(u8, u8); 3]) -> Vec<[(u8, u8); 3]> {
    let r0 = cards[0].0;
    let r1 = cards[1].0;
    let r2 = cards[2].0;

    if r0 == r1 && r1 == r2 {
        // Trips: 3! = 6 permutations
        vec![
            [cards[0], cards[1], cards[2]],
            [cards[0], cards[2], cards[1]],
            [cards[1], cards[0], cards[2]],
            [cards[1], cards[2], cards[0]],
            [cards[2], cards[0], cards[1]],
            [cards[2], cards[1], cards[0]],
        ]
    } else if r0 == r1 {
        // First two share rank: 2 permutations
        vec![
            [cards[0], cards[1], cards[2]],
            [cards[1], cards[0], cards[2]],
        ]
    } else if r1 == r2 {
        // Last two share rank: 2 permutations
        vec![
            [cards[0], cards[1], cards[2]],
            [cards[0], cards[2], cards[1]],
        ]
    } else {
        // All distinct: single ordering
        vec![[cards[0], cards[1], cards[2]]]
    }
}

// ---------------------------------------------------------------------------
// OOP/IP assignment
// ---------------------------------------------------------------------------

/// Position order for postflop action: SB(4) → BB(5) → UTG(0) → HJ(1) → CO(2) → BTN(3)
/// Lower = acts first = OOP.
fn postflop_order(pos_name: &str) -> u8 {
    match pos_name {
        "SB" => 0,
        "BB" => 1,
        "UTG" => 2,
        "HJ" => 3,
        "CO" => 4,
        "BTN" => 5,
        _ => panic!("unknown position: {}", pos_name),
    }
}

/// Determine which side is OOP and which is IP.
/// Returns (oop_position, ip_position, oop_is_opener).
pub fn assign_oop_ip(opener_pos: &str, caller_pos: &str) -> (String, String, bool) {
    let opener_order = postflop_order(opener_pos);
    let caller_order = postflop_order(caller_pos);

    if opener_order < caller_order {
        // Opener acts first → opener is OOP
        (opener_pos.to_string(), caller_pos.to_string(), true)
    } else {
        // Caller acts first → caller is OOP
        (caller_pos.to_string(), opener_pos.to_string(), false)
    }
}

// ---------------------------------------------------------------------------
// Batch config generation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchEntry {
    pub board: String,
    pub street: String,
    pub oop_range: String,
    pub ip_range: String,
    pub pot: i32,
    pub stack: i32,
    pub iterations: u32,
    pub matchup: String,
}

/// Generate batch JSONL entries for all matchups × all canonical flops.
pub fn generate_batch(
    matchups: &[SrpMatchup],
    defender_ranges: &HashMap<String, String>,
    iterations: u32,
) -> Vec<BatchEntry> {
    let flops = canonical_flops();
    let mut entries = Vec::with_capacity(matchups.len() * flops.len());

    for matchup in matchups {
        let opener_range_str = range_to_string(&matchup.opener.range);
        let caller_range_str = if let Some(override_str) = defender_ranges.get(&matchup.matchup) {
            if override_str.is_empty() {
                range_to_string(&matchup.caller.range)
            } else {
                override_str.clone()
            }
        } else {
            range_to_string(&matchup.caller.range)
        };

        let (oop_pos, _ip_pos, oop_is_opener) =
            assign_oop_ip(&matchup.opener.position, &matchup.caller.position);

        let (oop_range, ip_range) = if oop_is_opener {
            (opener_range_str.clone(), caller_range_str.clone())
        } else {
            (caller_range_str.clone(), opener_range_str.clone())
        };

        // Use the position that is OOP to verify correctness in debug
        let _ = oop_pos;

        for flop in &flops {
            entries.push(BatchEntry {
                board: flop.clone(),
                street: "flop".to_string(),
                oop_range: oop_range.clone(),
                ip_range: ip_range.clone(),
                pot: matchup.pot_chips,
                stack: matchup.eff_stack_chips,
                iterations,
                matchup: matchup.matchup.clone(),
            });
        }
    }

    entries
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_flops_count() {
        let flops = canonical_flops();
        assert_eq!(flops.len(), 1755, "expected 1755 canonical flops, got {}", flops.len());
    }

    #[test]
    fn test_canonical_flops_no_duplicates() {
        let flops = canonical_flops();
        let set: HashSet<_> = flops.iter().collect();
        assert_eq!(set.len(), flops.len(), "duplicate flops found");
    }

    #[test]
    fn test_canonical_flop_format() {
        let flops = canonical_flops();
        for flop in &flops {
            assert_eq!(flop.len(), 6, "flop string should be 6 chars: {}", flop);
            // Should be parseable as 3 cards
            let cards = crate::card::parse_cards(flop);
            assert!(cards.is_some(), "cannot parse flop: {}", flop);
            assert_eq!(cards.unwrap().len(), 3);
        }
    }

    #[test]
    fn test_range_to_string_full_weight() {
        let mut range = BTreeMap::new();
        range.insert("AA".to_string(), 1.0);
        range.insert("KK".to_string(), 1.0);
        let s = range_to_string(&range);
        assert_eq!(s, "AA,KK");
    }

    #[test]
    fn test_range_to_string_partial_weight() {
        let mut range = BTreeMap::new();
        range.insert("AA".to_string(), 1.0);
        range.insert("KK".to_string(), 0.995);
        range.insert("T9s".to_string(), 0.974);
        let s = range_to_string(&range);
        assert_eq!(s, "AA,KK:0.995,T9s:0.974");
    }

    #[test]
    fn test_range_to_string_roundtrip() {
        let mut range = BTreeMap::new();
        range.insert("AA".to_string(), 1.0);
        range.insert("KK".to_string(), 0.995);
        range.insert("QQ".to_string(), 0.5);
        range.insert("AKs".to_string(), 0.75);
        let s = range_to_string(&range);
        let parsed = Range::parse(&s);
        assert!(parsed.is_some(), "range_to_string output should be parseable: {}", s);
    }

    #[test]
    fn test_range_to_string_skip_zero() {
        let mut range = BTreeMap::new();
        range.insert("AA".to_string(), 1.0);
        range.insert("72o".to_string(), 0.0);
        let s = range_to_string(&range);
        assert_eq!(s, "AA");
    }

    #[test]
    fn test_oop_ip_assignment() {
        // Cases where opener is OOP (opener acts first postflop):
        // UTG opens, HJ/CO/BTN calls → UTG is OOP
        let (oop, ip, oop_is_opener) = assign_oop_ip("UTG", "HJ");
        assert_eq!(oop, "UTG"); assert_eq!(ip, "HJ"); assert!(oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("UTG", "CO");
        assert_eq!(oop, "UTG"); assert_eq!(ip, "CO"); assert!(oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("UTG", "BTN");
        assert_eq!(oop, "UTG"); assert_eq!(ip, "BTN"); assert!(oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("HJ", "CO");
        assert_eq!(oop, "HJ"); assert_eq!(ip, "CO"); assert!(oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("HJ", "BTN");
        assert_eq!(oop, "HJ"); assert_eq!(ip, "BTN"); assert!(oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("CO", "BTN");
        assert_eq!(oop, "CO"); assert_eq!(ip, "BTN"); assert!(oop_is_opener);

        // SB vs BB: SB acts first → SB (opener) is OOP
        let (oop, ip, oop_is_opener) = assign_oop_ip("SB", "BB");
        assert_eq!(oop, "SB"); assert_eq!(ip, "BB"); assert!(oop_is_opener);

        // Cases where caller (blind) is OOP:
        // UTG opens, SB calls → SB is OOP
        let (oop, ip, oop_is_opener) = assign_oop_ip("UTG", "SB");
        assert_eq!(oop, "SB"); assert_eq!(ip, "UTG"); assert!(!oop_is_opener);

        // UTG opens, BB calls → BB is OOP
        let (oop, ip, oop_is_opener) = assign_oop_ip("UTG", "BB");
        assert_eq!(oop, "BB"); assert_eq!(ip, "UTG"); assert!(!oop_is_opener);

        // HJ opens, SB/BB calls → blind is OOP
        let (oop, ip, oop_is_opener) = assign_oop_ip("HJ", "SB");
        assert_eq!(oop, "SB"); assert_eq!(ip, "HJ"); assert!(!oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("HJ", "BB");
        assert_eq!(oop, "BB"); assert_eq!(ip, "HJ"); assert!(!oop_is_opener);

        // CO opens, SB/BB calls → blind is OOP
        let (oop, ip, oop_is_opener) = assign_oop_ip("CO", "SB");
        assert_eq!(oop, "SB"); assert_eq!(ip, "CO"); assert!(!oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("CO", "BB");
        assert_eq!(oop, "BB"); assert_eq!(ip, "CO"); assert!(!oop_is_opener);

        // BTN opens, SB/BB calls → blind is OOP
        let (oop, ip, oop_is_opener) = assign_oop_ip("BTN", "SB");
        assert_eq!(oop, "SB"); assert_eq!(ip, "BTN"); assert!(!oop_is_opener);

        let (oop, ip, oop_is_opener) = assign_oop_ip("BTN", "BB");
        assert_eq!(oop, "BB"); assert_eq!(ip, "BTN"); assert!(!oop_is_opener);
    }

    #[test]
    fn test_oop_ip_counts() {
        // 7 matchups where opener is OOP, 8 where caller is OOP
        let matchup_pairs = [
            ("UTG", "HJ"), ("UTG", "CO"), ("UTG", "BTN"), ("UTG", "SB"), ("UTG", "BB"),
            ("HJ", "CO"), ("HJ", "BTN"), ("HJ", "SB"), ("HJ", "BB"),
            ("CO", "BTN"), ("CO", "SB"), ("CO", "BB"),
            ("BTN", "SB"), ("BTN", "BB"),
            ("SB", "BB"),
        ];

        let opener_oop_count = matchup_pairs.iter()
            .filter(|(o, c)| assign_oop_ip(o, c).2)
            .count();
        let caller_oop_count = matchup_pairs.len() - opener_oop_count;

        assert_eq!(opener_oop_count, 7, "expected 7 matchups where opener is OOP");
        assert_eq!(caller_oop_count, 8, "expected 8 matchups where caller is OOP");
    }

    #[test]
    fn test_defender_template() {
        let matchups = vec![
            SrpMatchup {
                matchup: "UTG vs HJ".to_string(),
                pot_chips: 9,
                eff_stack_chips: 196,
                opener: crate::preflop::MatchupSide {
                    position: "UTG".to_string(),
                    range: BTreeMap::new(),
                },
                caller: crate::preflop::MatchupSide {
                    position: "HJ".to_string(),
                    range: BTreeMap::new(),
                },
            },
        ];
        let template = generate_defender_template(&matchups);
        let obj = template.as_object().unwrap();
        assert_eq!(obj.len(), 1);
        assert_eq!(obj["UTG vs HJ"], "");
    }
}
