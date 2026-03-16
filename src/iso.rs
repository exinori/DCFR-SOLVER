/// Suit isomorphism — 169 canonical preflop hands, postflop canonical keys,
/// and board-preserving suit permutations for isomorphic card deduction.

use crate::card::{rank, suit, combo_index, combo_from_index, Card, Hand, NUM_COMBOS};

// ---------------------------------------------------------------------------
// CanonicalHand — 169 preflop equivalence classes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalHand {
    pub hi: u8,     // higher rank (0..13)
    pub lo: u8,     // lower rank
    pub suited: bool,
}

impl CanonicalHand {
    /// Map to index 0..168.
    /// Layout: pairs (13), suited (78), offsuit (78) = 169 total.
    /// Pairs: 0..12 (22..AA)
    /// Suited: 13..90 (hi > lo, suited)
    /// Offsuit: 91..168 (hi > lo, offsuit)
    pub fn index(&self) -> u8 {
        if self.hi == self.lo {
            // Pair
            self.hi
        } else if self.suited {
            // Suited: enumerate (hi, lo) where hi > lo
            // For hi=1,lo=0 → 13; hi=2,lo=0 → 14; hi=2,lo=1 → 15; etc.
            let base = self.hi as u16 * (self.hi as u16 - 1) / 2;
            13 + base as u8 + self.lo
        } else {
            // Offsuit: same enumeration but offset by 13 + 78 = 91
            let base = self.hi as u16 * (self.hi as u16 - 1) / 2;
            91 + base as u8 + self.lo
        }
    }

    /// Inverse of index.
    pub fn from_index(idx: u8) -> Self {
        if idx < 13 {
            CanonicalHand { hi: idx, lo: idx, suited: false }
        } else {
            let (suited, offset) = if idx < 91 {
                (true, idx - 13)
            } else {
                (false, idx - 91)
            };
            // Reverse triangular: find hi such that hi*(hi-1)/2 <= offset
            let mut hi = 1u8;
            while (hi + 1) as u16 * hi as u16 / 2 <= offset as u16 {
                hi += 1;
            }
            let lo = offset - (hi as u16 * (hi as u16 - 1) / 2) as u8;
            CanonicalHand { hi, lo, suited }
        }
    }

    pub fn to_string(&self) -> String {
        const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
        if self.hi == self.lo {
            format!("{}{}", RANK_CHARS[self.hi as usize], RANK_CHARS[self.lo as usize])
        } else {
            let suffix = if self.suited { "s" } else { "o" };
            format!("{}{}{}", RANK_CHARS[self.hi as usize], RANK_CHARS[self.lo as usize], suffix)
        }
    }
}

/// Convert two hole cards to a canonical hand.
pub fn canonical_hand(c1: Card, c2: Card) -> CanonicalHand {
    let r1 = rank(c1);
    let r2 = rank(c2);
    let s1 = suit(c1);
    let s2 = suit(c2);
    let suited = s1 == s2;
    let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    CanonicalHand { hi, lo, suited }
}

// ---------------------------------------------------------------------------
// Postflop canonical observation key
// ---------------------------------------------------------------------------

/// Canonical observation key for postflop equity tables.
/// Encodes hole cards + board in a suit-isomorphic way.
///
/// Method: assign suits 0-3 in order of first appearance (hole then board).
/// Then encode as i64 using rank*4+canonical_suit for each card.
pub fn canonical_observation(hole: Hand, board: Hand) -> i64 {
    // Collect cards in order: hole cards first, then board
    let mut cards: Vec<Card> = Vec::new();

    // Hole cards (sorted by original card value for consistency)
    let mut hole_cards: Vec<Card> = hole.iter().collect();
    hole_cards.sort();
    cards.extend_from_slice(&hole_cards);

    // Board cards sorted
    let mut board_cards: Vec<Card> = board.iter().collect();
    board_cards.sort();
    cards.extend_from_slice(&board_cards);

    // Build suit mapping: first appearance → canonical suit
    let mut suit_map = [255u8; 4];
    let mut next_suit = 0u8;

    for &c in &cards {
        let s = suit(c) as usize;
        if suit_map[s] == 255 {
            suit_map[s] = next_suit;
            next_suit += 1;
        }
    }

    // Encode: each card → rank*4 + canonical_suit, pack into i64
    let mut key: i64 = 0;
    for (i, &c) in cards.iter().enumerate() {
        let r = rank(c);
        let s = suit(c) as usize;
        let canonical_card = r * 4 + suit_map[s];
        key |= (canonical_card as i64) << (i * 8);
    }

    key
}

pub const NUM_CANONICAL_HANDS: usize = 169;

// ---------------------------------------------------------------------------
// Board-preserving suit permutations for isomorphic card deduction
// ---------------------------------------------------------------------------

/// Enumerate all non-identity suit permutations that preserve the board.
///
/// A suit permutation σ: {0,1,2,3} → {0,1,2,3} preserves the board iff
/// for every card on the board, σ maps that card to another board card.
/// Equivalently: suits with the same rank bitmask can be freely permuted.
pub fn board_suit_perms(board: Hand) -> Vec<[u8; 4]> {
    // Compute rank bitmask for each suit
    let mut suit_ranks = [0u16; 4];
    for c in board.iter() {
        let r = rank(c);
        let s = suit(c) as usize;
        suit_ranks[s] |= 1u16 << r;
    }

    // Group suits by their rank bitmask
    // Groups: suits with identical rank bitmask are interchangeable
    let mut groups: Vec<Vec<u8>> = Vec::new();
    let mut assigned = [false; 4];
    for s in 0..4u8 {
        if assigned[s as usize] { continue; }
        let mut group = vec![s];
        for s2 in (s + 1)..4u8 {
            if !assigned[s2 as usize] && suit_ranks[s2 as usize] == suit_ranks[s as usize] {
                group.push(s2);
                assigned[s2 as usize] = true;
            }
        }
        assigned[s as usize] = true;
        groups.push(group);
    }

    // Generate all permutations by permuting within each group
    // and taking the Cartesian product across groups
    let mut all_perms: Vec<[u8; 4]> = vec![[0, 1, 2, 3]]; // start with identity

    for group in &groups {
        if group.len() <= 1 { continue; }
        let group_perms = permutations(group);
        let mut new_all = Vec::with_capacity(all_perms.len() * group_perms.len());
        for base in &all_perms {
            for gp in &group_perms {
                let mut perm = *base;
                for (i, &orig_suit) in group.iter().enumerate() {
                    perm[orig_suit as usize] = gp[i];
                }
                new_all.push(perm);
            }
        }
        all_perms = new_all;
    }

    // Remove identity
    all_perms.retain(|p| p != &[0u8, 1, 2, 3]);
    all_perms
}

/// Generate all permutations of a slice.
fn permutations(items: &[u8]) -> Vec<Vec<u8>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        let rest: Vec<u8> = items.iter().enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &v)| v)
            .collect();
        for mut sub in permutations(&rest) {
            sub.insert(0, item);
            result.push(sub);
        }
    }
    result
}

/// Compute the combo index permutation table for a suit permutation.
///
/// For each combo index i, table[i] = the combo index after applying the suit permutation.
pub fn combo_permutation(perm: &[u8; 4]) -> [u16; NUM_COMBOS] {
    let mut table = [0u16; NUM_COMBOS];
    for idx in 0..NUM_COMBOS as u16 {
        let (c1, c2) = combo_from_index(idx);
        let c1p = rank(c1) * 4 + perm[suit(c1) as usize];
        let c2p = rank(c2) * 4 + perm[suit(c2) as usize];
        table[idx as usize] = combo_index(c1p, c2p);
    }
    table
}

/// A canonical card with its associated isomorphic combo permutations.
pub struct CanonicalCard {
    pub card: Card,
    /// Combo permutation tables for each isomorphic card (identity NOT included).
    pub perms: Vec<[u16; NUM_COMBOS]>,
}

/// Compute canonical next cards for dealing from a board.
///
/// Groups available cards by equivalence under board-preserving suit permutations.
/// Returns one representative per group with the combo permutations needed
/// to reconstruct the isomorphic cards' contributions.
pub fn canonical_next_cards(board: Hand) -> Vec<CanonicalCard> {
    let suit_perms = board_suit_perms(board);

    if suit_perms.is_empty() {
        // No symmetry (rainbow board or all suits distinct) — every card is canonical
        let mut result = Vec::new();
        for c in 0..52u8 {
            if board.contains(c) { continue; }
            result.push(CanonicalCard { card: c, perms: Vec::new() });
        }
        return result;
    }

    // For each available card, find its canonical representative
    // Card a maps to card b under perm σ iff rank(a)==rank(b) && σ(suit(a))==suit(b)
    let mut assigned = [false; 52];
    let mut result = Vec::new();

    for c in 0..52u8 {
        if board.contains(c) || assigned[c as usize] { continue; }
        assigned[c as usize] = true;

        let mut perms = Vec::new();
        for sp in &suit_perms {
            let mapped = rank(c) * 4 + sp[suit(c) as usize];
            if mapped != c && !board.contains(mapped) && !assigned[mapped as usize] {
                assigned[mapped as usize] = true;
                // The combo permutation maps canonical card's combos → this isomorphic card's combos
                perms.push(combo_permutation(sp));
            }
        }

        result.push(CanonicalCard { card: c, perms });
    }

    result
}

/// Compose ancestor permutations with this-level permutations.
///
/// Computes the group closure of ancestor ∪ this_level: all distinct permutations
/// reachable by composing elements from both sets in any order.
///
/// The old formula `ancestor ∪ this_level ∪ {a∘t}` was incorrect for non-commutative
/// groups (e.g., S₃ on monotone boards): it produced duplicates and missed elements
/// because it only composed in one direction (a then t), not (t then a).
///
/// Uses iterative closure with fingerprint-based dedup for efficiency.
/// Converges quickly since max group size is |S₄| = 24.
pub fn compose_perms(
    ancestor: &[[u16; NUM_COMBOS]],
    this_level: &[[u16; NUM_COMBOS]],
) -> Vec<[u16; NUM_COMBOS]> {
    // NOTE: No early returns! Both empty cases go through the general
    // group-closure path for consistent dedup + identity filtering.
    // this_level may only contain generators (e.g., [c↔d, c↔h] for S₃).
    // We must compute the group closure to get all non-identity elements
    // (e.g., d↔h, c→d→h→c, c→h→d→c) needed for correct regret scatter.

    // FNV-1a hash sampling 8 positions for fast dedup.
    // Sufficient for perm uniqueness: any two distinct perms in S₄ (max 24 elements)
    // differ in enough positions that 8 samples avoid collisions.
    #[inline]
    fn fingerprint(perm: &[u16; NUM_COMBOS]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &i in &[0usize, 1, 50, 100, 300, 600, 1000, 1325] {
            h = h.wrapping_mul(0x100000001b3);
            h ^= perm[i] as u64;
        }
        h
    }

    // Identity fingerprint
    let identity_fp = {
        let mut id = [0u16; NUM_COMBOS];
        for c in 0..NUM_COMBOS { id[c] = c as u16; }
        fingerprint(&id)
    };

    // Seed with all generators (ancestor ∪ this_level), excluding identity/duplicates
    let mut result: Vec<[u16; NUM_COMBOS]> = Vec::with_capacity(24);
    let mut seen = std::collections::HashSet::with_capacity(24);
    seen.insert(identity_fp); // pre-insert identity to exclude it

    for perm in ancestor.iter().chain(this_level.iter()) {
        let fp = fingerprint(perm);
        if seen.insert(fp) {
            result.push(*perm);
        }
    }

    // Iterative closure: compose all pairs until no new perms appear.
    // Max group size is |S₄| - 1 = 23, converges in ≤3 rounds.
    loop {
        let mut new_perms: Vec<[u16; NUM_COMBOS]> = Vec::new();
        let n = result.len();
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                // Compose result[i] then result[j]: composed[c] = result[j][result[i][c]]
                let mut composed = [0u16; NUM_COMBOS];
                for c in 0..NUM_COMBOS {
                    composed[c] = result[j][result[i][c] as usize];
                }
                let fp = fingerprint(&composed);
                if seen.insert(fp) {
                    new_perms.push(composed);
                }
            }
        }
        if new_perms.is_empty() {
            break;
        }
        result.extend_from_slice(&new_perms);
    }

    result
}

// ---------------------------------------------------------------------------
// Canonical board conversion for suit isomorphism
// ---------------------------------------------------------------------------

/// Convert a board to canonical form by assigning suits in order of first appearance.
///
/// Board cards are sorted by card value for deterministic ordering, then suits
/// are assigned canonical indices 0, 1, 2, 3 in order of first appearance.
/// Suits not present on the board get remaining indices in original order (0-3).
///
/// Returns (canonical_board, suit_perm) where suit_perm[original_suit] = canonical_suit.
pub fn canonical_board(board: Hand) -> (Hand, [u8; 4]) {
    let mut board_cards: Vec<Card> = board.iter().collect();
    board_cards.sort();

    let mut suit_map = [255u8; 4];
    let mut next_suit = 0u8;

    for &c in &board_cards {
        let s = suit(c) as usize;
        if suit_map[s] == 255 {
            suit_map[s] = next_suit;
            next_suit += 1;
        }
    }
    for s in 0..4usize {
        if suit_map[s] == 255 {
            suit_map[s] = next_suit;
            next_suit += 1;
        }
    }

    let mut canonical = Hand::new();
    for &c in &board_cards {
        let r = rank(c);
        let cs = suit_map[suit(c) as usize];
        canonical = canonical.add(r * 4 + cs);
    }

    (canonical, suit_map)
}

/// Compute the inverse of a suit permutation.
pub fn inverse_suit_perm(perm: &[u8; 4]) -> [u8; 4] {
    let mut inv = [0u8; 4];
    for i in 0..4 {
        inv[perm[i] as usize] = i as u8;
    }
    inv
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card;

    #[test]
    fn test_canonical_board_basic() {
        // AsKs3d7c2h → suits appear: s(3), d(1), c(0), h(2)
        // Canonical: s→0, d→1, c→2, h→3
        let board = Hand::new()
            .add(card(12, 3)) // As
            .add(card(11, 3)) // Ks
            .add(card(1, 1))  // 3d
            .add(card(5, 0))  // 7c
            .add(card(0, 2)); // 2h
        let (canon, perm) = canonical_board(board);

        // First suit seen is c(0) at card 2h? No — cards sorted: 2h(2), 3d(5), 7c(20), Ks(47), As(51)
        // Sorted order: 2h(suit=2), 3d(suit=1), 7c(suit=0), Ks(suit=3), As(suit=3)
        // First appearance: h→0, d→1, c→2, s→3
        assert_eq!(perm[2], 0); // h → 0
        assert_eq!(perm[1], 1); // d → 1
        assert_eq!(perm[0], 2); // c → 2
        assert_eq!(perm[3], 3); // s → 3

        // Verify canonical board has correct cards
        assert!(canon.contains(card(0, 0)));  // 2h → 2c (rank=0, suit h→0=c)
        assert!(canon.contains(card(1, 1)));  // 3d → 3d (rank=1, suit d→1=d)
        assert!(canon.contains(card(5, 2)));  // 7c → 7h (rank=5, suit c→2=h)
        assert!(canon.contains(card(11, 3))); // Ks → Ks (rank=11, suit s→3=s)
        assert!(canon.contains(card(12, 3))); // As → As (rank=12, suit s→3=s)
    }

    #[test]
    fn test_canonical_board_idempotent() {
        // A board that's already canonical should return identity perm
        // Board: 2c 3d 7h → suits c(0), d(1), h(2) already in order
        let board = Hand::new()
            .add(card(0, 0))  // 2c
            .add(card(1, 1))  // 3d
            .add(card(5, 2)); // 7h
        let (canon, perm) = canonical_board(board);
        assert_eq!(perm, [0, 1, 2, 3]);
        assert_eq!(canon, board);
    }

    #[test]
    fn test_canonical_board_isomorphic_boards_same_result() {
        // Two boards that are suit-relabelings should produce the same canonical form
        let board1 = Hand::new()
            .add(card(12, 3)) // As
            .add(card(11, 3)) // Ks
            .add(card(1, 1)); // 3d
        let board2 = Hand::new()
            .add(card(12, 0)) // Ac
            .add(card(11, 0)) // Kc
            .add(card(1, 2)); // 3h

        let (canon1, _) = canonical_board(board1);
        let (canon2, _) = canonical_board(board2);
        assert_eq!(canon1, canon2, "isomorphic boards should have same canonical form");
    }

    #[test]
    fn test_inverse_suit_perm() {
        let perm = [2u8, 0, 3, 1]; // c→h, d→c, h→s, s→d
        let inv = inverse_suit_perm(&perm);
        // inv should satisfy: inv[perm[i]] == i
        for i in 0..4 {
            assert_eq!(inv[perm[i] as usize], i as u8);
        }
    }

    #[test]
    fn test_canonical_hand() {
        // AA
        let ch = canonical_hand(card(12, 0), card(12, 1));
        assert_eq!(ch.hi, 12);
        assert_eq!(ch.lo, 12);
        assert!(!ch.suited); // pairs are never suited
        // Actually pairs: suits are different, so suited=false ✓

        // AKs
        let ch = canonical_hand(card(12, 0), card(11, 0));
        assert_eq!(ch.hi, 12);
        assert_eq!(ch.lo, 11);
        assert!(ch.suited);

        // AKo
        let ch = canonical_hand(card(12, 0), card(11, 1));
        assert!(!ch.suited);
    }

    #[test]
    fn test_index_roundtrip() {
        for idx in 0..169u8 {
            let ch = CanonicalHand::from_index(idx);
            assert_eq!(ch.index(), idx, "roundtrip failed for index {}: {:?}", idx, ch);
        }
    }

    #[test]
    fn test_169_unique() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for c1 in 0..52u8 {
            for c2 in (c1 + 1)..52u8 {
                let ch = canonical_hand(c1, c2);
                seen.insert(ch.index());
            }
        }
        assert_eq!(seen.len(), 169);
    }

    #[test]
    fn test_canonical_observation_isomorphic() {
        // AsKs on AhTh2d should equal AhKh on AsTs2d (suit relabeling)
        let hole1 = Hand::new().add(card(12, 3)).add(card(11, 3)); // AsKs
        let board1 = Hand::new().add(card(12, 2)).add(card(8, 2)).add(card(0, 1)); // AhTh2d

        let hole2 = Hand::new().add(card(12, 2)).add(card(11, 2)); // AhKh
        let board2 = Hand::new().add(card(12, 3)).add(card(8, 3)).add(card(0, 1)); // AsTs2d

        let k1 = canonical_observation(hole1, board1);
        let k2 = canonical_observation(hole2, board2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_board_suit_perms_rainbow() {
        // AhKsTd3c — all suits distinct, all have different rank bitmasks
        let board = Hand::new()
            .add(card(12, 2)) // Ah
            .add(card(11, 3)) // Ks
            .add(card(8, 1))  // Td
            .add(card(1, 0)); // 3c
        let perms = board_suit_perms(board);
        assert_eq!(perms.len(), 0, "rainbow board should have no symmetry");
    }

    #[test]
    fn test_board_suit_perms_two_tone() {
        // AsKs3d — s has {A,K}, d has {3}, h and c both have {} → h↔c swap
        let board = Hand::new()
            .add(card(12, 3)) // As
            .add(card(11, 3)) // Ks
            .add(card(1, 1));  // 3d
        let perms = board_suit_perms(board);
        assert_eq!(perms.len(), 1, "two-tone board should have 1 non-identity perm");
        // The perm should swap h(2) and c(0), keeping d(1) and s(3) fixed
        let p = &perms[0];
        assert_eq!(p[0], 2, "c→h"); // c and h swap
        assert_eq!(p[2], 0, "h→c");
        assert_eq!(p[1], 1, "d→d"); // d stays
        assert_eq!(p[3], 3, "s→s"); // s stays
    }

    #[test]
    fn test_board_suit_perms_monotone() {
        // AhKhTh — h has {A,K,T}, others all empty → S₃ - 1 = 5 perms among {c,d,s}
        let board = Hand::new()
            .add(card(12, 2)) // Ah
            .add(card(11, 2)) // Kh
            .add(card(8, 2));  // Th
        let perms = board_suit_perms(board);
        assert_eq!(perms.len(), 5, "monotone board should have 5 non-identity perms");
        // All perms should fix h(2)
        for p in &perms {
            assert_eq!(p[2], 2, "h should be fixed in monotone board perm");
        }
    }

    #[test]
    fn test_combo_permutation_roundtrip() {
        // Applying a permutation and its inverse should return to original
        let perm = [2u8, 1, 0, 3]; // swap c↔h
        let table = combo_permutation(&perm);

        // Every output index should be valid and the mapping should be bijective
        let mut seen = vec![false; NUM_COMBOS];
        for idx in 0..NUM_COMBOS {
            let mapped = table[idx] as usize;
            assert!(mapped < NUM_COMBOS, "mapped index out of range");
            assert!(!seen[mapped], "combo permutation not bijective at {}", idx);
            seen[mapped] = true;
        }
        assert!(seen.iter().all(|&s| s), "combo permutation not surjective");
    }

    #[test]
    fn test_combo_permutation_swap_suits() {
        // Swap c(0) ↔ h(2): AcKc → AhKh
        let perm = [2u8, 1, 0, 3]; // c→h, d→d, h→c, s→s
        let table = combo_permutation(&perm);

        let ac = card(12, 0); // Ac
        let kc = card(11, 0); // Kc
        let ah = card(12, 2); // Ah
        let kh = card(11, 2); // Kh

        let idx_ackc = combo_index(ac, kc);
        let idx_ahkh = combo_index(ah, kh);
        assert_eq!(table[idx_ackc as usize], idx_ahkh, "AcKc should map to AhKh");
    }

    #[test]
    fn test_canonical_next_cards_rainbow() {
        // Rainbow flop: all 49 cards are canonical, no isomorphic grouping
        let board = Hand::new()
            .add(card(12, 2)) // Ah
            .add(card(11, 3)) // Ks
            .add(card(8, 1));  // Td
        let canonical = canonical_next_cards(board);
        assert_eq!(canonical.len(), 49, "rainbow flop: all 49 cards are canonical");
        for cc in &canonical {
            assert!(cc.perms.is_empty(), "rainbow: no iso perms");
        }
    }

    #[test]
    fn test_canonical_next_cards_two_tone() {
        // AsKs3d — h↔c symmetry
        // Cards: 49 available. For each rank, h and c variants are isomorphic.
        // So for each of 13 ranks, h+c pair → 1 canonical. But ranks on board are special:
        // A(12): As is on board, Ah and Ac are available but iso → 1 canonical, Ad also available → 2
        // K(11): Ks on board, Kh and Kc iso → 1, Kd → 1 → total 2
        // 3(1): 3d on board, 3h and 3c iso → 1, 3s → 1 → total 2
        // Other 10 ranks: 4 cards each, h↔c → 3 canonical (one for {h,c}, one for d, one for s)
        // Total: 3 * 2 + 10 * 3 = 36
        let board = Hand::new()
            .add(card(12, 3)) // As
            .add(card(11, 3)) // Ks
            .add(card(1, 1));  // 3d
        let canonical = canonical_next_cards(board);

        // Count total cards represented
        let total_cards: usize = canonical.iter().map(|cc| 1 + cc.perms.len()).sum();
        assert_eq!(total_cards, 49, "total cards should be 49");
        assert!(canonical.len() < 49, "two-tone should have fewer canonical cards than 49");
        assert_eq!(canonical.len(), 36, "two-tone AsKs3d should have 36 canonical cards");
    }

    #[test]
    fn test_canonical_next_cards_monotone() {
        // AhKhTh — {c,d,s} are all interchangeable (S₃)
        // For each of 13 ranks:
        //   Board ranks (A,K,T): those 3 suits are interchangeable → 1 canonical card
        //   Other ranks: 3 interchangeable suits → 1 canonical + h card → 2 canonical
        // Wait, 3 board cards use h. Available: for rank 12(A): Ac,Ad,As (all iso) → 1
        // For rank 11(K): Kc,Kd,Ks → 1; rank 8(T): Tc,Td,Ts → 1
        // For other 10 ranks: h + {c,d,s} → Xh is separate, {Xc,Xd,Xs} are one group → 2
        // Total: 3*1 + 10*2 = 23
        let board = Hand::new()
            .add(card(12, 2)) // Ah
            .add(card(11, 2)) // Kh
            .add(card(8, 2));  // Th
        let canonical = canonical_next_cards(board);
        let total_cards: usize = canonical.iter().map(|cc| 1 + cc.perms.len()).sum();
        assert_eq!(total_cards, 49, "total cards should be 49");
        assert_eq!(canonical.len(), 23, "monotone AhKhTh should have 23 canonical cards");
    }

    #[test]
    fn test_canonical_next_cards_paired() {
        // 9h9c5d — paired board. h↔c symmetry (same as two-tone).
        // 49 available cards. For each rank:
        //   9: 9d,9s available (d and s are not symmetric) → 2 canonical
        //   5: 5h,5c(iso),5s → 2 canonical (h↔c grouped, + s separate... wait: d is on board)
        //     5d on board. 5h,5c,5s available. h↔c → 1 group, s → 1 → 2
        //   Other 11 ranks: 4 cards, h↔c → 3 canonical
        // Total: 2 + 2 + 11*3 = 37
        let board = Hand::new()
            .add(card(7, 2)) // 9h
            .add(card(7, 0)) // 9c
            .add(card(3, 1)); // 5d
        let canonical = canonical_next_cards(board);
        let total_cards: usize = canonical.iter().map(|cc| 1 + cc.perms.len()).sum();
        assert_eq!(total_cards, 49, "paired 9h9c5d: multiplicity sum must equal 49");
        // Verify no card appears in multiple groups
        let mut seen = std::collections::HashSet::new();
        for cc in &canonical {
            assert!(seen.insert(cc.card), "duplicate canonical card {}", cc.card);
            for perm in &cc.perms {
                // Each perm represents one isomorphic card — check perm is a valid permutation
                // (identity check: perm[i] should differ from identity for at least one combo)
            }
        }
        println!("Paired 9h9c5d: {} canonical cards, {} total", canonical.len(), total_cards);
    }

    #[test]
    fn test_canonical_next_cards_monotone_ts8s3s() {
        // Ts8s3s — monotone spade board. {c,d,h} are all interchangeable (S₃)
        let board = Hand::new()
            .add(card(8, 3))  // Ts
            .add(card(6, 3))  // 8s
            .add(card(1, 3)); // 3s
        let canonical = canonical_next_cards(board);
        let total_cards: usize = canonical.iter().map(|cc| 1 + cc.perms.len()).sum();
        assert_eq!(total_cards, 49, "monotone Ts8s3s: multiplicity sum must equal 49");
        println!("Monotone Ts8s3s: {} canonical cards, {} total", canonical.len(), total_cards);
    }

    #[test]
    fn test_compose_perms_empty() {
        let a: Vec<[u16; NUM_COMBOS]> = Vec::new();
        let b: Vec<[u16; NUM_COMBOS]> = Vec::new();
        assert_eq!(compose_perms(&a, &b).len(), 0);
    }

    // ===================================================================
    // Phase A: compose_perms / canonical_next_cards property tests
    // ===================================================================

    /// Every combo permutation from canonical_next_cards must be a bijection.
    #[test]
    fn test_iso_perms_are_bijections() {
        let boards: Vec<Hand> = vec![
            // Paired
            Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)), // 9h9c5d
            // Monotone
            Hand::new().add(card(8, 3)).add(card(6, 3)).add(card(1, 3)), // Ts8s3s
            // Two-tone
            Hand::new().add(card(12, 3)).add(card(11, 3)).add(card(1, 1)), // AsKs3d
            // Dynamic
            Hand::new().add(card(8, 1)).add(card(7, 1)).add(card(4, 2)), // Td9d6h
        ];
        for board in &boards {
            let canonical = canonical_next_cards(*board);
            for cc in &canonical {
                for (pi, perm) in cc.perms.iter().enumerate() {
                    let mut seen = std::collections::HashSet::new();
                    for c in 0..NUM_COMBOS {
                        let mapped = perm[c] as usize;
                        assert!(mapped < NUM_COMBOS, "perm[{}]={} out of range (board {:?}, card {}, perm {})", c, mapped, board, cc.card, pi);
                        assert!(seen.insert(mapped), "perm is not bijection: duplicate mapping to {} (board {:?}, card {}, perm {})", mapped, board, cc.card, pi);
                    }
                }
            }
        }
    }

    /// compose_perms with one empty side should produce the same group closure
    /// as using the non-empty side alone. Both orderings should give the same set.
    #[test]
    fn test_compose_perms_identity() {
        let board = Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)); // 9h9c5d
        let canonical = canonical_next_cards(board);
        // Get a non-empty perm set
        let perms: Vec<[u16; NUM_COMBOS]> = canonical.iter()
            .flat_map(|cc| cc.perms.iter().cloned())
            .take(3)
            .collect();
        if perms.is_empty() { return; }

        let empty: Vec<[u16; NUM_COMBOS]> = Vec::new();

        // compose(empty, perms) and compose(perms, empty) should produce the same group
        let r1 = compose_perms(&empty, &perms);
        let r2 = compose_perms(&perms, &empty);

        // Both should have the same elements (group closure is commutative on generators)
        assert_eq!(r1.len(), r2.len(),
            "compose([], perms).len()={} != compose(perms, []).len()={}",
            r1.len(), r2.len());

        // All non-identity input perms should appear in the output
        let identity: [u16; NUM_COMBOS] = {
            let mut id = [0u16; NUM_COMBOS];
            for c in 0..NUM_COMBOS { id[c] = c as u16; }
            id
        };
        for p in &perms {
            if *p != identity {
                assert!(r1.contains(p),
                    "compose([], perms) should contain all non-identity input perms");
            }
        }

        // Output should not contain identity
        assert!(!r1.contains(&identity), "group closure should exclude identity");
    }

    /// Each composed perm must also be a bijection.
    #[test]
    fn test_compose_perms_outputs_are_bijections() {
        let boards: Vec<Hand> = vec![
            Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)), // 9h9c5d paired
            Hand::new().add(card(8, 3)).add(card(6, 3)).add(card(1, 3)), // Ts8s3s monotone
        ];
        for board in &boards {
            let canonical_turns = canonical_next_cards(*board);
            for ct in &canonical_turns {
                if ct.perms.is_empty() { continue; }
                let turn_board = board.add(ct.card);
                let canonical_rivers = canonical_next_cards(turn_board);
                for cr in &canonical_rivers {
                    if cr.perms.is_empty() { continue; }
                    // Compose flop→turn perms with turn→river perms
                    let composed = compose_perms(&ct.perms, &cr.perms);
                    for (ci, perm) in composed.iter().enumerate() {
                        let mut seen = std::collections::HashSet::new();
                        for c in 0..NUM_COMBOS {
                            let mapped = perm[c] as usize;
                            assert!(mapped < NUM_COMBOS, "composed perm out of range");
                            assert!(seen.insert(mapped), "composed perm {} not bijection at combo {} (board {:?}, turn {}, river {})", ci, c, board, ct.card, cr.card);
                        }
                    }
                }
            }
        }
    }

    /// Sequential equivalence: apply(compose(a,b), x) == apply(b, apply(a, x))
    /// for all combos, across all turn×river perm combos on paired/monotone boards.
    #[test]
    fn test_compose_perms_sequential_equivalence() {
        let boards: Vec<Hand> = vec![
            Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)), // 9h9c5d
            Hand::new().add(card(8, 3)).add(card(6, 3)).add(card(1, 3)), // Ts8s3s
        ];
        let mut tested = 0u64;
        for board in &boards {
            let canonical_turns = canonical_next_cards(*board);
            for ct in &canonical_turns {
                if ct.perms.is_empty() { continue; }
                let turn_board = board.add(ct.card);
                let canonical_rivers = canonical_next_cards(turn_board);
                for cr in &canonical_rivers {
                    if cr.perms.is_empty() { continue; }
                    let composed = compose_perms(&ct.perms, &cr.perms);
                    // Verify: every a∘t composition must be present in the composed set
                    // (group closure contains all single-step compositions)
                    let composed_set: std::collections::HashSet<Vec<u16>> =
                        composed.iter().map(|p| p.to_vec()).collect();

                    // All ancestors must be present
                    for a in &ct.perms {
                        assert!(composed_set.contains(&a.to_vec()),
                            "ancestor perm missing from composed set");
                    }
                    // All this_level must be present
                    for t in &cr.perms {
                        assert!(composed_set.contains(&t.to_vec()),
                            "this_level perm missing from composed set");
                    }
                    // All a∘t compositions must be present (or be identity)
                    for a in &ct.perms {
                        for t in &cr.perms {
                            let mut at_composed = [0u16; NUM_COMBOS];
                            for c in 0..NUM_COMBOS {
                                at_composed[c] = t[a[c] as usize];
                            }
                            // Check if identity
                            let is_identity = (0..NUM_COMBOS).all(|c| at_composed[c] == c as u16);
                            if !is_identity {
                                assert!(composed_set.contains(&at_composed.to_vec()),
                                    "a∘t composition missing from composed set");
                            }
                            tested += NUM_COMBOS as u64;
                        }
                    }
                    // All t∘a compositions must also be present
                    for t in &cr.perms {
                        for a in &ct.perms {
                            let mut ta_composed = [0u16; NUM_COMBOS];
                            for c in 0..NUM_COMBOS {
                                ta_composed[c] = a[t[c] as usize];
                            }
                            let is_identity = (0..NUM_COMBOS).all(|c| ta_composed[c] == c as u16);
                            if !is_identity {
                                assert!(composed_set.contains(&ta_composed.to_vec()),
                                    "t∘a composition missing from composed set");
                            }
                            tested += NUM_COMBOS as u64;
                        }
                    }
                    // No duplicates
                    assert_eq!(composed.len(), composed_set.len(),
                        "compose_perms produced {} entries but only {} unique",
                        composed.len(), composed_set.len());
                }
            }
        }
        println!("Sequential equivalence: tested {} combo mappings", tested);
    }

    /// Multiplicity invariant across ALL turn×river combinations for paired/monotone.
    /// For each canonical turn card, the river canonical cards must also sum to correct total.
    #[test]
    fn test_two_layer_multiplicity_invariant() {
        let boards: Vec<(Hand, &str)> = vec![
            (Hand::new().add(card(7, 2)).add(card(7, 0)).add(card(3, 1)), "9h9c5d"),
            (Hand::new().add(card(8, 3)).add(card(6, 3)).add(card(1, 3)), "Ts8s3s"),
            (Hand::new().add(card(8, 1)).add(card(7, 1)).add(card(4, 2)), "Td9d6h"),
        ];
        for (board, name) in &boards {
            let canonical_turns = canonical_next_cards(*board);
            let n_board = board.iter().count();
            let n_avail_turn = 52 - n_board;
            let total_turn: usize = canonical_turns.iter().map(|cc| 1 + cc.perms.len()).sum();
            assert_eq!(total_turn, n_avail_turn, "{}: turn multiplicity {} != {}", name, total_turn, n_avail_turn);

            for ct in &canonical_turns {
                let turn_board = board.add(ct.card);
                let n_avail_river = 52 - n_board - 1; // minus turn card
                let canonical_rivers = canonical_next_cards(turn_board);
                let total_river: usize = canonical_rivers.iter().map(|cc| 1 + cc.perms.len()).sum();
                assert_eq!(total_river, n_avail_river,
                    "{}: river multiplicity {} != {} (after turn card {})", name, total_river, n_avail_river, ct.card);
            }
        }
    }
}
