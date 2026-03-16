/// Deck — shuffled card source with seeded RNG.

use crate::card::{Card, Hand, NUM_CARDS};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub struct Deck {
    cards: Vec<Card>,
    index: usize,
}

impl Deck {
    /// Create a shuffled deck with a given seed.
    pub fn new_seeded(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut cards: Vec<Card> = (0..NUM_CARDS as Card).collect();
        cards.shuffle(&mut rng);
        Deck { cards, index: 0 }
    }

    /// Create a shuffled deck from an existing RNG.
    pub fn new_from_rng(rng: &mut SmallRng) -> Self {
        let mut cards: Vec<Card> = (0..NUM_CARDS as Card).collect();
        cards.shuffle(rng);
        Deck { cards, index: 0 }
    }

    /// Draw one card.
    pub fn draw(&mut self) -> Card {
        let c = self.cards[self.index];
        self.index += 1;
        c
    }

    /// Draw n cards as a Hand.
    pub fn draw_hand(&mut self, n: usize) -> Hand {
        let mut h = Hand::new();
        for _ in 0..n {
            h = h.add(self.draw());
        }
        h
    }

    /// Draw one card that doesn't conflict with dead cards.
    pub fn draw_excluding(&mut self, dead: Hand) -> Card {
        loop {
            let c = self.draw();
            if !dead.contains(c) {
                return c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deck_draw() {
        let mut deck = Deck::new_seeded(42);
        let h = deck.draw_hand(5);
        assert_eq!(h.count(), 5);
    }

    #[test]
    fn test_deck_deterministic() {
        let mut d1 = Deck::new_seeded(123);
        let mut d2 = Deck::new_seeded(123);
        for _ in 0..52 {
            assert_eq!(d1.draw(), d2.draw());
        }
    }
}
