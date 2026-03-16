//! Card representation and combinatorial utilities.
//!
//! Card: u8 where value = rank * 4 + suit
//!   rank 0..13 (2,3,...,A), suit 0..4 (c,d,h,s)
//! Hand: u64 bitmask of cards
//! CombIter: Gosper's hack for C(n,k) enumeration
//! combo_index / combo_from_index: canonical 2-card combo indexing (0..1325)

use std::fmt;

// ---------------------------------------------------------------------------
// Card
// ---------------------------------------------------------------------------

pub type Card = u8;

pub const NUM_CARDS: usize = 52;

#[inline]
pub fn rank(c: Card) -> u8 {
    c >> 2
}

#[inline]
pub fn suit(c: Card) -> u8 {
    c & 3
}

#[inline]
pub fn card(rank: u8, suit: u8) -> Card {
    rank * 4 + suit
}

/// Rank names: 2=0 .. A=12
const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

pub fn card_to_string(c: Card) -> String {
    let r = rank(c) as usize;
    let s = suit(c) as usize;
    format!("{}{}", RANK_CHARS[r], SUIT_CHARS[s])
}

pub fn parse_card(s: &str) -> Option<Card> {
    let bytes: Vec<char> = s.chars().collect();
    if bytes.len() != 2 {
        return None;
    }
    let r = RANK_CHARS.iter().position(|&c| c == bytes[0].to_ascii_uppercase())?;
    let su = SUIT_CHARS.iter().position(|&c| c == bytes[1].to_ascii_lowercase())?;
    Some(card(r as u8, su as u8))
}

pub fn parse_cards(s: &str) -> Option<Vec<Card>> {
    let s = s.trim();
    if s.is_empty() {
        return Some(vec![]);
    }
    let mut cards = Vec::new();
    let mut chars = s.chars().peekable();
    while chars.peek().is_some() {
        let r = chars.next()?;
        let su = chars.next()?;
        let cs = format!("{}{}", r, su);
        cards.push(parse_card(&cs)?);
    }
    Some(cards)
}

// ---------------------------------------------------------------------------
// Hand (u64 bitmask)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Hand(pub u64);

impl Hand {
    pub const EMPTY: Hand = Hand(0);

    #[inline]
    pub fn new() -> Self {
        Hand(0)
    }

    #[inline]
    pub fn from_card(c: Card) -> Self {
        Hand(1u64 << c)
    }

    #[inline]
    pub fn add(self, c: Card) -> Self {
        Hand(self.0 | (1u64 << c))
    }

    #[inline]
    pub fn remove(self, c: Card) -> Self {
        Hand(self.0 & !(1u64 << c))
    }

    #[inline]
    pub fn contains(self, c: Card) -> bool {
        (self.0 >> c) & 1 != 0
    }

    #[inline]
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub fn overlaps(self, other: Hand) -> bool {
        self.0 & other.0 != 0
    }

    #[inline]
    pub fn union(self, other: Hand) -> Self {
        Hand(self.0 | other.0)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Iterate over set card bits.
    pub fn iter(self) -> HandIter {
        HandIter(self.0)
    }
}

impl fmt::Debug for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cards: Vec<String> = self.iter().map(card_to_string).collect();
        write!(f, "[{}]", cards.join(" "))
    }
}

pub struct HandIter(u64);

impl Iterator for HandIter {
    type Item = Card;
    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            None
        } else {
            let bit = self.0.trailing_zeros() as Card;
            self.0 &= self.0 - 1; // clear lowest bit
            Some(bit)
        }
    }
}

// ---------------------------------------------------------------------------
// CombIter — Gosper's hack for C(n, k)
// ---------------------------------------------------------------------------

/// Iterates all k-bit submasks of an n-bit universe via Gosper's hack.
/// Usage: CombIter::new(n, k) iterates C(n,k) bitmasks.
pub struct CombIter {
    bits: u64,
    limit: u64,
}

impl CombIter {
    pub fn new(n: u32, k: u32) -> Self {
        if k == 0 || k > n {
            return CombIter {
                bits: 0,
                limit: 0,
            };
        }
        let bits = (1u64 << k) - 1;
        let limit = 1u64 << n;
        CombIter { bits, limit }
    }
}

impl Iterator for CombIter {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.bits >= self.limit {
            return None;
        }
        let result = self.bits;
        if self.bits == 0 {
            self.bits = self.limit; // stop iteration for k=0 edge case
            return Some(result);
        }
        // Gosper's hack
        let c = self.bits & (!self.bits).wrapping_add(1); // lowest set bit
        let r = self.bits + c;
        self.bits = (((r ^ self.bits) >> 2) / c) | r;
        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Combo indexing: map 2-card combos to 0..1325
// ---------------------------------------------------------------------------

/// Maps two cards (c1 < c2) to a unique index in 0..1325.
/// Standard triangular indexing: index = c2*(c2-1)/2 + c1
#[inline]
pub fn combo_index(c1: Card, c2: Card) -> u16 {
    let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
    (hi as u16 * (hi as u16 - 1) / 2) + lo as u16
}

/// Inverse of combo_index — O(1) via precomputed lookup table.
#[inline]
pub fn combo_from_index(idx: u16) -> (Card, Card) {
    unsafe { *COMBO_TABLE.get_unchecked(idx as usize) }
}

pub const NUM_COMBOS: usize = 1326; // C(52, 2)

/// Precomputed combo decomposition table [1326].
static COMBO_TABLE: [(Card, Card); NUM_COMBOS] = {
    let mut table = [(0u8, 0u8); NUM_COMBOS];
    let mut hi: u16 = 1;
    let mut base: u16 = 0; // hi*(hi-1)/2
    let mut idx: usize = 0;
    while idx < NUM_COMBOS {
        while base + hi <= idx as u16 {
            base += hi;
            hi += 1;
        }
        let lo = idx as u16 - base;
        table[idx] = (lo as Card, hi as Card);
        idx += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// Card-to-combo mapping: for each card, which combo indices contain it.
// ---------------------------------------------------------------------------

/// For each card (0..52), the 51 combo indices that contain that card.
/// Used to efficiently zero out reach for dealt cards in chance nodes.
pub static CARD_COMBOS: [[u16; 51]; 52] = {
    let mut table = [[0u16; 51]; 52];
    let mut card: u8 = 0;
    while card < 52 {
        let mut count: usize = 0;
        let mut other: u8 = 0;
        while other < 52 {
            if other != card {
                let (lo, hi) = if card < other { (card, other) } else { (other, card) };
                let idx = (hi as u16 * (hi as u16 - 1) / 2) + lo as u16;
                table[card as usize][count] = idx;
                count += 1;
            }
            other += 1;
        }
        card += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_basics() {
        let ace_spades = card(12, 3); // As
        assert_eq!(rank(ace_spades), 12);
        assert_eq!(suit(ace_spades), 3);
        assert_eq!(card_to_string(ace_spades), "As");

        let two_clubs = card(0, 0);
        assert_eq!(card_to_string(two_clubs), "2c");
    }

    #[test]
    fn test_parse_card() {
        assert_eq!(parse_card("As"), Some(card(12, 3)));
        assert_eq!(parse_card("2c"), Some(card(0, 0)));
        assert_eq!(parse_card("Th"), Some(card(8, 2)));
        assert_eq!(parse_card("Kd"), Some(card(11, 1)));
    }

    #[test]
    fn test_parse_cards() {
        let cards = parse_cards("AsKhTd").unwrap();
        assert_eq!(cards.len(), 3);
        assert_eq!(cards[0], card(12, 3));
        assert_eq!(cards[1], card(11, 2));
        assert_eq!(cards[2], card(8, 1));
    }

    #[test]
    fn test_hand() {
        let mut h = Hand::new();
        h = h.add(card(12, 3)); // As
        h = h.add(card(11, 2)); // Kh
        assert!(h.contains(card(12, 3)));
        assert!(!h.contains(card(0, 0)));
        assert_eq!(h.count(), 2);

        h = h.remove(card(12, 3));
        assert!(!h.contains(card(12, 3)));
        assert_eq!(h.count(), 1);
    }

    #[test]
    fn test_hand_iter() {
        let h = Hand::new().add(0).add(10).add(51);
        let cards: Vec<Card> = h.iter().collect();
        assert_eq!(cards, vec![0, 10, 51]);
    }

    #[test]
    fn test_comb_iter() {
        // C(5,2) = 10
        let count = CombIter::new(5, 2).count();
        assert_eq!(count, 10);

        // C(52,2) = 1326
        let count = CombIter::new(52, 2).count();
        assert_eq!(count, 1326);

        // C(52,5) = 2598960
        let count = CombIter::new(52, 5).count();
        assert_eq!(count, 2_598_960);
    }

    #[test]
    fn test_combo_index_roundtrip() {
        for idx in 0..NUM_COMBOS as u16 {
            let (c1, c2) = combo_from_index(idx);
            assert!(c1 < c2);
            assert_eq!(combo_index(c1, c2), idx);
        }
    }

    #[test]
    fn test_combo_index_covers_all() {
        let mut seen = vec![false; NUM_COMBOS];
        for c1 in 0..52u8 {
            for c2 in (c1 + 1)..52u8 {
                let idx = combo_index(c1, c2) as usize;
                assert!(!seen[idx], "duplicate index {}", idx);
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_card_combos() {
        // Each card should map to exactly 51 combos
        for card in 0..52u8 {
            let combos = &CARD_COMBOS[card as usize];
            assert_eq!(combos.len(), 51);
            for &ci in combos {
                let (c1, c2) = combo_from_index(ci);
                assert!(c1 == card || c2 == card, "combo {} should contain card {}", ci, card);
            }
        }
    }
}
