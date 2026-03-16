/// Range — 1326-combo weight vector for subgame solving.

use crate::card::{combo_from_index, combo_index, parse_card, Hand, NUM_COMBOS};
use crate::iso::{canonical_hand, CanonicalHand};

#[derive(Clone)]
pub struct Range {
    pub weights: [f32; NUM_COMBOS],
}

impl Range {
    /// All combos at weight 1.0.
    pub fn uniform() -> Self {
        Range {
            weights: [1.0; NUM_COMBOS],
        }
    }

    /// All combos at weight 0.0.
    pub fn empty() -> Self {
        Range {
            weights: [0.0; NUM_COMBOS],
        }
    }

    /// Set blockers to 0 (any combo containing a dead card).
    pub fn remove_blockers(&mut self, dead: Hand) {
        for i in 0..NUM_COMBOS {
            let (c1, c2) = combo_from_index(i as u16);
            if dead.contains(c1) || dead.contains(c2) {
                self.weights[i] = 0.0;
            }
        }
    }

    /// Number of live (non-zero) combos.
    pub fn count_live(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }

    /// Total weight sum.
    pub fn total_weight(&self) -> f32 {
        self.weights.iter().sum()
    }

    /// Build range from canonical hand weights.
    pub fn from_canonical(hand_weights: &[(CanonicalHand, f32)]) -> Self {
        let mut range = Range::empty();
        for &(ch, weight) in hand_weights {
            if weight <= 0.0 {
                continue;
            }
            // Find all combos matching this canonical hand
            for c1 in 0..52u8 {
                for c2 in (c1 + 1)..52u8 {
                    let ch2 = canonical_hand(c1, c2);
                    if ch2 == ch {
                        let idx = combo_index(c1, c2) as usize;
                        range.weights[idx] = weight;
                    }
                }
            }
        }
        range
    }

    /// Parse range string like "AA:1,AKs:0.5,QQ-TT:1,KQo:0.75"
    /// Also supports GTO+ format: "[82.0]QJs,Q4s,A9o[/82.0]" (82% weight)
    pub fn parse(s: &str) -> Option<Self> {
        let mut range = Range::empty();

        // Pre-process: expand GTO+ bracket notation into weight:hand pairs.
        // "[82.0]QJs,Q4s[/82.0]" → "QJs:0.82,Q4s:0.82"
        let expanded = expand_gtoplus_brackets(s);
        let input = expanded.as_deref().unwrap_or(s);

        for part in input.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            // Split weight
            let (hand_str, weight) = if let Some(idx) = part.find(':') {
                let w: f32 = part[idx + 1..].trim().parse().ok()?;
                (part[..idx].trim(), w)
            } else {
                (part, 1.0f32)
            };

            // Check for range notation (e.g., "QQ-TT")
            if let Some(dash_idx) = hand_str.find('-') {
                let hi_str = &hand_str[..dash_idx];
                let lo_str = &hand_str[dash_idx + 1..];
                let hands = expand_range(hi_str, lo_str)?;
                for ch in hands {
                    add_canonical(&mut range, ch, weight);
                }
            } else if let Some(ch) = parse_canonical_hand(hand_str) {
                add_canonical(&mut range, ch, weight);
            } else if hand_str.len() == 4 {
                // Specific combo: "Ac9s" → card1=Ac, card2=9s
                let c1 = parse_card(&hand_str[..2])?;
                let c2 = parse_card(&hand_str[2..])?;
                let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                range.weights[combo_index(lo, hi) as usize] = weight;
            } else {
                return None;
            }
        }
        Some(range)
    }
}

/// Expand GTO+ bracket notation: "[82.0]QJs,Q4s[/82.0]" → "QJs:0.82,Q4s:0.82"
/// Returns None if no brackets found (input can be used as-is).
fn expand_gtoplus_brackets(s: &str) -> Option<String> {
    if !s.contains('[') {
        return None;
    }
    let mut result = String::with_capacity(s.len() * 2);
    let mut i = 0;
    let bytes = s.as_bytes();
    let len = bytes.len();

    while i < len {
        if bytes[i] == b'[' && i + 1 < len && bytes[i + 1] != b'/' {
            // Parse weight: [XX.XX]
            let close = s[i..].find(']')?;
            let weight_str = &s[i + 1..i + close];
            let pct: f32 = weight_str.parse().ok()?;
            let weight = pct / 100.0;
            i += close + 1; // skip past ']'

            // Find closing tag [/XX.XX]
            let close_tag = format!("[/{}]", weight_str);
            let end = s[i..].find(&close_tag)?;
            let hands_str = &s[i..i + end];
            i += end + close_tag.len();

            // Expand: each hand in hands_str gets the weight
            for hand in hands_str.split(',') {
                let hand = hand.trim();
                if hand.is_empty() { continue; }
                if !result.is_empty() { result.push(','); }
                result.push_str(hand);
                result.push(':');
                result.push_str(&format!("{:.4}", weight));
            }
        } else {
            // Regular character — copy through
            let next_bracket = s[i..].find('[').unwrap_or(len - i);
            let chunk = &s[i..i + next_bracket];
            if !result.is_empty() && !chunk.is_empty() && !chunk.starts_with(',') && !result.ends_with(',') {
                result.push(',');
            }
            result.push_str(chunk);
            i += next_bracket;
        }
    }
    Some(result)
}

fn add_canonical(range: &mut Range, ch: CanonicalHand, weight: f32) {
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            if canonical_hand(c1, c2) == ch {
                range.weights[combo_index(c1, c2) as usize] = weight;
            }
        }
    }
}

const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

fn parse_rank(c: char) -> Option<u8> {
    RANK_CHARS.iter().position(|&r| r == c.to_ascii_uppercase()).map(|r| r as u8)
}

fn parse_canonical_hand(s: &str) -> Option<CanonicalHand> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 || chars.len() > 3 {
        return None;
    }
    let r1 = parse_rank(chars[0])?;
    let r2 = parse_rank(chars[1])?;
    let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };

    let suited = if chars.len() == 3 {
        match chars[2].to_ascii_lowercase() {
            's' => true,
            'o' => false,
            _ => return None,
        }
    } else {
        // Pair (no suffix) or both ranks same
        if hi == lo {
            false // pairs
        } else {
            return None; // ambiguous: need s/o suffix
        }
    };

    Some(CanonicalHand { hi, lo, suited })
}

fn expand_range(hi_str: &str, lo_str: &str) -> Option<Vec<CanonicalHand>> {
    let hi = parse_canonical_hand(hi_str)?;
    let lo = parse_canonical_hand(lo_str)?;

    let mut hands = Vec::new();

    if hi.hi == hi.lo && lo.hi == lo.lo {
        // Pair range: e.g. QQ-TT
        let top = hi.hi;
        let bot = lo.hi;
        for r in bot..=top {
            hands.push(CanonicalHand { hi: r, lo: r, suited: false });
        }
    } else if hi.suited == lo.suited && hi.hi == lo.hi {
        // Same high card, range of kickers: e.g. ATs-A5s
        let suit = hi.suited;
        let high = hi.hi;
        let top_lo = hi.lo;
        let bot_lo = lo.lo;
        for r in bot_lo..=top_lo {
            hands.push(CanonicalHand { hi: high, lo: r, suited: suit });
        }
    } else {
        // Single expansion
        hands.push(hi);
        hands.push(lo);
    }

    Some(hands)
}

impl std::fmt::Debug for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let live = self.count_live();
        write!(f, "Range({} live combos)", live)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform() {
        let r = Range::uniform();
        assert_eq!(r.count_live(), 1326);
    }

    #[test]
    fn test_blockers() {
        let mut r = Range::uniform();
        let dead = Hand::new().add(0).add(1); // 2c, 2d dead
        r.remove_blockers(dead);
        // Each card blocks 51 combos, but 2c-2d combo counted twice
        // Blocked = 51 + 51 - 1 = 101
        assert_eq!(r.count_live(), 1326 - 101);
    }

    #[test]
    fn test_parse_pair() {
        let r = Range::parse("AA").unwrap();
        // AA = C(4,2) = 6 combos
        assert_eq!(r.count_live(), 6);
    }

    #[test]
    fn test_parse_suited() {
        let r = Range::parse("AKs").unwrap();
        // AKs = 4 combos (one per suit)
        assert_eq!(r.count_live(), 4);
    }

    #[test]
    fn test_parse_offsuit() {
        let r = Range::parse("AKo").unwrap();
        // AKo = 12 combos
        assert_eq!(r.count_live(), 12);
    }

    #[test]
    fn test_parse_weight() {
        let r = Range::parse("AA:0.5").unwrap();
        assert_eq!(r.count_live(), 6);
        let (c1, c2) = (crate::card::card(12, 0), crate::card::card(12, 1));
        let idx = combo_index(c1, c2) as usize;
        assert!((r.weights[idx] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_range() {
        let r = Range::parse("QQ-TT").unwrap();
        // QQ + JJ + TT = 6+6+6 = 18 combos
        assert_eq!(r.count_live(), 18);
    }

    #[test]
    fn test_parse_complex() {
        let r = Range::parse("AA:1,AKs:0.5,QQ-TT:1").unwrap();
        // AA=6, AKs=4, QQ=6, JJ=6, TT=6 = 28 combos
        assert_eq!(r.count_live(), 28);
    }

    #[test]
    fn test_parse_gtoplus_brackets() {
        // GTO+ format: [82.0]QJs[/82.0] means 82% weight
        let r = Range::parse("AA,[82.0]QJs[/82.0]").unwrap();
        // AA = 6 combos at 1.0, QJs = 4 combos at 0.82
        assert_eq!(r.count_live(), 10);
        // Check QJs weight: Q=10, J=9, suited → e.g. QsJs = card(10,0) card(9,0)
        let qs = crate::card::card(10, 0);
        let js = crate::card::card(9, 0);
        let idx = combo_index(qs, js) as usize;
        assert!((r.weights[idx] - 0.82).abs() < 0.01, "QsJs weight={}", r.weights[idx]);
    }

    #[test]
    fn test_parse_gtoplus_multi_hands() {
        // Multiple hands in one bracket
        let r = Range::parse("[50.0]AA,KK[/50.0]").unwrap();
        // AA=6, KK=6, all at 0.50
        assert_eq!(r.count_live(), 12);
        let ac = crate::card::card(12, 0);
        let ad = crate::card::card(12, 1);
        let idx = combo_index(ac, ad) as usize;
        assert!((r.weights[idx] - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_parse_gtoplus_real_range() {
        // Snippet from actual GTO+ BTN open range
        let s = "AA-33,AKs,[77.0]22[/77.0],[76.0]96s[/76.0]";
        let r = Range::parse(s).unwrap();
        // AA-33 = 12 pairs * 6 = 72 combos, AKs = 4, 22 = 6 at 0.77, 96s = 4 at 0.76
        assert_eq!(r.count_live(), 72 + 4 + 6 + 4);
    }
}
