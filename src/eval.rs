/// Hand evaluator using bit operations only (no lookup tables).
///
/// Evaluates the best 5-card hand from 5-7 cards.
/// Strength: (HandRank, kickers) — fully ordered for comparison.

use crate::card::{rank, suit, Hand};

// ---------------------------------------------------------------------------
// HandRank
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandRank {
    HighCard = 0,
    OnePair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
}

// ---------------------------------------------------------------------------
// Strength — compact comparable hand value
// ---------------------------------------------------------------------------

/// Strength encodes hand rank + up to 5 kicker ranks in a single u32.
/// Format: [4 bits hand_rank][4 bits k1][4 bits k2][4 bits k3][4 bits k4][4 bits k5]
/// This allows direct comparison via Ord.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Strength(pub u32);

impl Strength {
    fn new(hand_rank: HandRank, kickers: &[u8]) -> Self {
        let mut val = (hand_rank as u32) << 20;
        for (i, &k) in kickers.iter().take(5).enumerate() {
            val |= (k as u32) << (16 - i * 4);
        }
        Strength(val)
    }

    pub fn hand_rank(self) -> HandRank {
        match self.0 >> 20 {
            0 => HandRank::HighCard,
            1 => HandRank::OnePair,
            2 => HandRank::TwoPair,
            3 => HandRank::ThreeOfAKind,
            4 => HandRank::Straight,
            5 => HandRank::Flush,
            6 => HandRank::FullHouse,
            7 => HandRank::FourOfAKind,
            8 => HandRank::StraightFlush,
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// evaluate — main entry point
// ---------------------------------------------------------------------------

/// Evaluate the best 5-card hand from a Hand containing 5-7 cards.
pub fn evaluate(hand: Hand) -> Strength {
    let n = hand.count();
    debug_assert!((5..=7).contains(&n), "evaluate requires 5-7 cards, got {}", n);

    // Extract rank counts and per-suit rank masks
    let mut rank_count = [0u8; 13];
    let mut suit_mask = [0u16; 4]; // bitmask of ranks per suit

    for c in hand.iter() {
        let r = rank(c) as usize;
        let s = suit(c) as usize;
        rank_count[r] += 1;
        suit_mask[s] |= 1 << r;
    }

    // Build overall rank bitmask
    let rank_mask: u16 = rank_count
        .iter()
        .enumerate()
        .fold(0u16, |acc, (r, &cnt)| if cnt > 0 { acc | (1 << r) } else { acc });

    // Check flush (5+ cards of same suit)
    let flush_suit = suit_mask.iter().position(|&m| m.count_ones() >= 5);

    // Check straight (using rank bitmask)
    let straight_high = find_straight(rank_mask);

    // Straight flush check
    if let Some(fs) = flush_suit {
        let flush_straight = find_straight(suit_mask[fs]);
        if let Some(high) = flush_straight {
            return Strength::new(HandRank::StraightFlush, &[high]);
        }
    }

    // Four of a kind
    if let Some(quad_rank) = find_n_of_kind(&rank_count, 4) {
        let kicker = highest_rank_excluding(&rank_count, &[quad_rank], 1);
        return Strength::new(HandRank::FourOfAKind, &[quad_rank, kicker[0]]);
    }

    // Full house: highest trips + highest pair (from remaining)
    let trips: Vec<u8> = all_n_of_kind(&rank_count, 3);
    if !trips.is_empty() {
        let best_trip = trips[0]; // highest first
        // For pair: any other trips rank also counts as pair, plus actual pairs
        let mut pair_candidates: Vec<u8> = Vec::new();
        for &t in &trips {
            if t != best_trip {
                pair_candidates.push(t);
            }
        }
        for r in (0..13).rev() {
            if rank_count[r as usize] == 2 {
                pair_candidates.push(r);
            }
        }
        if !pair_candidates.is_empty() {
            return Strength::new(HandRank::FullHouse, &[best_trip, pair_candidates[0]]);
        }
    }

    // Flush
    if let Some(fs) = flush_suit {
        let flush_kickers = top_n_bits(suit_mask[fs], 5);
        return Strength::new(HandRank::Flush, &flush_kickers);
    }

    // Straight
    if let Some(high) = straight_high {
        return Strength::new(HandRank::Straight, &[high]);
    }

    // Three of a kind (no full house)
    if !trips.is_empty() {
        let kickers = highest_rank_excluding(&rank_count, &[trips[0]], 2);
        return Strength::new(HandRank::ThreeOfAKind, &[trips[0], kickers[0], kickers[1]]);
    }

    // Two pair / One pair
    let pairs: Vec<u8> = all_n_of_kind(&rank_count, 2);
    if pairs.len() >= 2 {
        let kickers = highest_rank_excluding(&rank_count, &[pairs[0], pairs[1]], 1);
        return Strength::new(HandRank::TwoPair, &[pairs[0], pairs[1], kickers[0]]);
    }
    if pairs.len() == 1 {
        let kickers = highest_rank_excluding(&rank_count, &[pairs[0]], 3);
        return Strength::new(
            HandRank::OnePair,
            &[pairs[0], kickers[0], kickers[1], kickers[2]],
        );
    }

    // High card
    let kickers = top_n_ranks(&rank_count, 5);
    Strength::new(HandRank::HighCard, &kickers)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the highest straight in a rank bitmask. Returns high card rank.
/// Handles A-2-3-4-5 (wheel) with high=3 (rank of 5).
fn find_straight(mask: u16) -> Option<u8> {
    // Check from A-high (12) down to 6-high (4)
    for high in (4..=12).rev() {
        let pattern = 0x1Fu16 << (high - 4);
        if mask & pattern == pattern {
            return Some(high as u8);
        }
    }
    // Wheel: A,2,3,4,5 = bits 12,0,1,2,3
    let wheel: u16 = (1 << 12) | 0xF;
    if mask & wheel == wheel {
        return Some(3); // 5-high straight
    }
    None
}

/// Find highest rank with exactly n of that rank.
fn find_n_of_kind(rank_count: &[u8; 13], n: u8) -> Option<u8> {
    for r in (0..13).rev() {
        if rank_count[r] == n {
            return Some(r as u8);
        }
    }
    None
}

/// All ranks with exactly n of that rank, sorted descending.
fn all_n_of_kind(rank_count: &[u8; 13], n: u8) -> Vec<u8> {
    let mut result = Vec::new();
    for r in (0..13).rev() {
        if rank_count[r] == n {
            result.push(r as u8);
        }
    }
    result
}

/// Get `count` highest ranks excluding specified ranks.
fn highest_rank_excluding(rank_count: &[u8; 13], exclude: &[u8], count: usize) -> Vec<u8> {
    let mut result = Vec::new();
    for r in (0..13).rev() {
        if result.len() >= count {
            break;
        }
        if rank_count[r as usize] > 0 && !exclude.contains(&(r as u8)) {
            result.push(r as u8);
        }
    }
    result
}

/// Top n ranks by count > 0, descending.
fn top_n_ranks(rank_count: &[u8; 13], n: usize) -> Vec<u8> {
    let mut result = Vec::new();
    for r in (0..13).rev() {
        if result.len() >= n {
            break;
        }
        if rank_count[r as usize] > 0 {
            result.push(r as u8);
        }
    }
    result
}

/// Top n bits from a bitmask, returned as rank values descending.
fn top_n_bits(mask: u16, n: usize) -> Vec<u8> {
    let mut result = Vec::new();
    for r in (0..13).rev() {
        if result.len() >= n {
            break;
        }
        if mask & (1 << r) != 0 {
            result.push(r as u8);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{card, Hand};

    fn make_hand(cards: &[Card]) -> Hand {
        let mut h = Hand::new();
        for &c in cards {
            h = h.add(c);
        }
        h
    }

    use crate::card::Card;

    #[test]
    fn test_high_card() {
        // 2c 4d 6h 8s Tc — high card T
        let h = make_hand(&[card(0, 0), card(2, 1), card(4, 2), card(6, 3), card(8, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::HighCard);
    }

    #[test]
    fn test_one_pair() {
        // Ac Ad 3h 5s 7c
        let h = make_hand(&[card(12, 0), card(12, 1), card(1, 2), card(3, 3), card(5, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::OnePair);
    }

    #[test]
    fn test_two_pair() {
        // Ac Ad Kh Ks 7c
        let h = make_hand(&[card(12, 0), card(12, 1), card(11, 2), card(11, 3), card(5, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::TwoPair);
    }

    #[test]
    fn test_three_of_kind() {
        // Ac Ad Ah 5s 7c
        let h = make_hand(&[card(12, 0), card(12, 1), card(12, 2), card(3, 3), card(5, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::ThreeOfAKind);
    }

    #[test]
    fn test_straight() {
        // 5c 6d 7h 8s 9c
        let h = make_hand(&[card(3, 0), card(4, 1), card(5, 2), card(6, 3), card(7, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::Straight);
    }

    #[test]
    fn test_wheel() {
        // Ac 2d 3h 4s 5c
        let h = make_hand(&[card(12, 0), card(0, 1), card(1, 2), card(2, 3), card(3, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::Straight);
        // Wheel should be lower than 6-high straight
        let h2 = make_hand(&[card(1, 0), card(2, 1), card(3, 2), card(4, 3), card(5, 0)]);
        let s2 = evaluate(h2);
        assert!(s2 > s, "6-high straight should beat wheel");
    }

    #[test]
    fn test_flush() {
        // 2c 4c 6c 8c Tc — all clubs
        let h = make_hand(&[card(0, 0), card(2, 0), card(4, 0), card(6, 0), card(8, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::Flush);
    }

    #[test]
    fn test_full_house() {
        // Ac Ad Ah Ks Kd
        let h = make_hand(&[card(12, 0), card(12, 1), card(12, 2), card(11, 3), card(11, 1)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::FullHouse);
    }

    #[test]
    fn test_four_of_kind() {
        // Ac Ad Ah As Kc
        let h = make_hand(&[card(12, 0), card(12, 1), card(12, 2), card(12, 3), card(11, 0)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::FourOfAKind);
    }

    #[test]
    fn test_straight_flush() {
        // 5s 6s 7s 8s 9s
        let h = make_hand(&[card(3, 3), card(4, 3), card(5, 3), card(6, 3), card(7, 3)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::StraightFlush);
    }

    #[test]
    fn test_royal_flush() {
        // Ts Js Qs Ks As
        let h = make_hand(&[card(8, 3), card(9, 3), card(10, 3), card(11, 3), card(12, 3)]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::StraightFlush);
    }

    #[test]
    fn test_seven_card_hand() {
        // 7-card: should find best 5
        // Board: 2c 3c 4c 5c 6c (straight flush)
        // Hole: As Kd
        let h = make_hand(&[
            card(0, 0),  // 2c
            card(1, 0),  // 3c
            card(2, 0),  // 4c
            card(3, 0),  // 5c
            card(4, 0),  // 6c
            card(12, 3), // As
            card(11, 1), // Kd
        ]);
        let s = evaluate(h);
        assert_eq!(s.hand_rank(), HandRank::StraightFlush);
    }

    #[test]
    fn test_ordering() {
        // SF > 4K > FH > Flush > Straight > 3K > 2P > 1P > HC
        let sf = make_hand(&[card(3, 3), card(4, 3), card(5, 3), card(6, 3), card(7, 3)]);
        let fk = make_hand(&[card(12, 0), card(12, 1), card(12, 2), card(12, 3), card(11, 0)]);
        let fh = make_hand(&[card(12, 0), card(12, 1), card(12, 2), card(11, 3), card(11, 1)]);
        let fl = make_hand(&[card(0, 0), card(2, 0), card(4, 0), card(6, 0), card(8, 0)]);
        let st = make_hand(&[card(3, 0), card(4, 1), card(5, 2), card(6, 3), card(7, 0)]);
        let tk = make_hand(&[card(12, 0), card(12, 1), card(12, 2), card(3, 3), card(5, 0)]);
        let tp = make_hand(&[card(12, 0), card(12, 1), card(11, 2), card(11, 3), card(5, 0)]);
        let op = make_hand(&[card(12, 0), card(12, 1), card(1, 2), card(3, 3), card(5, 0)]);
        let hc = make_hand(&[card(0, 0), card(2, 1), card(4, 2), card(6, 3), card(8, 0)]);

        let strengths: Vec<Strength> = vec![sf, fk, fh, fl, st, tk, tp, op, hc]
            .into_iter()
            .map(evaluate)
            .collect();

        for i in 0..strengths.len() - 1 {
            assert!(
                strengths[i] > strengths[i + 1],
                "{:?} should beat {:?}",
                strengths[i],
                strengths[i + 1]
            );
        }
    }

    #[test]
    fn test_kicker_comparison() {
        // AK vs AQ high card
        let ak = make_hand(&[card(12, 0), card(11, 1), card(5, 2), card(3, 3), card(1, 0)]);
        let aq = make_hand(&[card(12, 0), card(10, 1), card(5, 2), card(3, 3), card(1, 0)]);
        assert!(evaluate(ak) > evaluate(aq));
    }
}
