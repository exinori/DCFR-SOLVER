/// Neural network value function for depth-limited solving.
///
/// Wraps an ONNX model that predicts per-combo EV for a given
/// (board, ranges, pot, stacks, street) state. Used to replace
/// CFR subtree traversal at depth boundaries.

#[cfg(feature = "nn")]
use ort::session::Session;
#[cfg(feature = "nn")]
use std::sync::Mutex;

use crate::card::{Hand, NUM_COMBOS};
use crate::game::{Player, Street};
use crate::range::Range;

/// Number of input features per sample.
/// board: 52 (binary), ranges: 1326*2 (raw weights), pot: 1, stack: 1, street: 2, total_reach: 2 = 2710
pub const INPUT_DIM: usize = 52 + NUM_COMBOS * 2 + 1 + 1 + 2 + 2;
/// Output: per-combo EV for OOP and IP = 1326 * 2
pub const OUTPUT_DIM: usize = NUM_COMBOS * 2;

/// Value network for depth-limited solving.
///
/// The NN predicts normalized counterfactual values (CFVs).
/// Training normalization: EV_oop / (pot × total_ip_reach),
///                         EV_ip  / (pot × total_oop_reach).
/// During inference, we denormalize by multiplying back:
///   raw_CFV_oop = NN_output × pot × total_ip_reach
///   raw_CFV_ip  = NN_output × pot × total_oop_reach
#[cfg(feature = "nn")]
pub struct ValueNet {
    session: Mutex<Session>,
}

#[cfg(feature = "nn")]
impl ValueNet {
    /// Load ONNX model from file.
    pub fn load(path: &str) -> Self {
        let session = Session::builder()
            .expect("Failed to create ONNX session builder")
            .with_intra_threads(1)
            .expect("Failed to set intra threads")
            .commit_from_file(path)
            .unwrap_or_else(|e| panic!("Failed to load ONNX model from {}: {}", path, e));
        ValueNet { session: Mutex::new(session) }
    }

    /// Predict per-combo CFVs for a single state.
    ///
    /// Returns [f32; NUM_COMBOS] of per-combo EV (not reach-weighted)
    /// for the given traverser, in original combo index space.
    /// Denormalizes the NN output by multiplying by pot × total_opp_reach.
    pub fn predict(
        &self,
        board: Hand,
        ranges: &[Range; 2],
        pot: i32,
        stacks: [i32; 2],
        street: Street,
        traverser: Player,
    ) -> [f32; NUM_COMBOS] {
        let input = encode_input(board, ranges, pot, stacks, street);

        // ort 2.0 API: Tensor::from_array takes (shape, data) tuple
        let input_tensor = ort::value::Tensor::from_array(
            ([1usize, INPUT_DIM], input.into_boxed_slice())
        ).expect("Failed to create input tensor");

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![input_tensor])
            .expect("ONNX inference failed");

        // try_extract_tensor returns (&Shape, &[f32])
        let (_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract output tensor");

        // Denormalize: NN was trained on EV / (pot × total_opp_reach)
        let opp = 1 - traverser as usize;
        let total_opp_reach: f32 = ranges[opp].weights.iter().sum();
        let scale = pot as f32 * total_opp_reach.max(1.0);

        let mut ev = [0.0f32; NUM_COMBOS];
        let offset = if traverser == 0 { 0 } else { NUM_COMBOS };
        for c in 0..NUM_COMBOS {
            ev[c] = output_data[offset + c] * scale;
        }
        ev
    }

    /// Batch predict for multiple boards, returning EVs for BOTH players.
    ///
    /// Single NN forward pass per batch. Returns Vec of [[OOP EVs]; [IP EVs]]
    /// in original combo index space, denormalized by pot × total_opp_reach.
    /// Used for pre-computing depth-limited leaf values before CFR iterations.
    pub fn predict_batch_both(
        &self,
        inputs: &[(Hand, [Range; 2], i32, [i32; 2], Street)],
    ) -> Vec<[[f32; NUM_COMBOS]; 2]> {
        let batch_size = inputs.len();
        if batch_size == 0 { return Vec::new(); }

        let mut flat_input = Vec::with_capacity(batch_size * INPUT_DIM);
        for (board, ranges, pot, stacks, street) in inputs {
            flat_input.extend_from_slice(&encode_input(*board, ranges, *pot, *stacks, *street));
        }

        let input_tensor = ort::value::Tensor::from_array(
            ([batch_size, INPUT_DIM], flat_input.into_boxed_slice())
        ).expect("Failed to create batch input tensor");

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![input_tensor])
            .expect("ONNX batch inference failed");

        let (_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract batch output tensor");

        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let row_offset = b * OUTPUT_DIM;
            let pot_f = inputs[b].2 as f32;

            // OOP: denormalize by pot × total_ip_reach
            let total_ip_reach: f32 = inputs[b].1[1].weights.iter().sum();
            let scale_oop = pot_f * total_ip_reach.max(1.0);

            // IP: denormalize by pot × total_oop_reach
            let total_oop_reach: f32 = inputs[b].1[0].weights.iter().sum();
            let scale_ip = pot_f * total_oop_reach.max(1.0);

            let mut ev = [[0.0f32; NUM_COMBOS]; 2];
            for c in 0..NUM_COMBOS {
                ev[0][c] = output_data[row_offset + c] * scale_oop;
                ev[1][c] = output_data[row_offset + NUM_COMBOS + c] * scale_ip;
            }
            results.push(ev);
        }
        results
    }

    /// Batch predict for multiple boards (e.g., all turn/river cards at a chance node).
    ///
    /// Takes a slice of (board, ranges, pot, stacks, street) and returns
    /// per-combo EVs for each. Used at chance nodes where 45-47 cards are dealt.
    pub fn predict_batch(
        &self,
        inputs: &[(Hand, [Range; 2], i32, [i32; 2], Street)],
        traverser: Player,
    ) -> Vec<[f32; NUM_COMBOS]> {
        let batch_size = inputs.len();
        let mut flat_input = Vec::with_capacity(batch_size * INPUT_DIM);
        for (board, ranges, pot, stacks, street) in inputs {
            flat_input.extend_from_slice(&encode_input(*board, ranges, *pot, *stacks, *street));
        }

        let input_tensor = ort::value::Tensor::from_array(
            ([batch_size, INPUT_DIM], flat_input.into_boxed_slice())
        ).expect("Failed to create batch input tensor");

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![input_tensor])
            .expect("ONNX batch inference failed");

        let (_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract batch output tensor");

        let offset = if traverser == 0 { 0 } else { NUM_COMBOS };
        let opp = 1 - traverser as usize;
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let pot_f = inputs[b].2 as f32;
            let total_opp_reach: f32 = inputs[b].1[opp].weights.iter().sum();
            let scale = pot_f * total_opp_reach.max(1.0);
            let mut ev = [0.0f32; NUM_COMBOS];
            let row_offset = b * OUTPUT_DIM;
            for c in 0..NUM_COMBOS {
                ev[c] = output_data[row_offset + offset + c] * scale;
            }
            results.push(ev);
        }
        results
    }
}

/// Encode a game state into a flat feature vector for the NN.
pub fn encode_input(
    board: Hand,
    ranges: &[Range; 2],
    pot: i32,
    stacks: [i32; 2],
    street: Street,
) -> Vec<f32> {
    let mut features = Vec::with_capacity(INPUT_DIM);

    // Board: 52-dim binary (which cards are on the board)
    for c in 0..52u8 {
        features.push(if board.contains(c) { 1.0 } else { 0.0 });
    }

    // Ranges: 1326 * 2 floats, NORMALIZED to sum=1.0 per player
    let total_oop: f32 = ranges[0].weights.iter().sum::<f32>().max(1e-8);
    let total_ip: f32 = ranges[1].weights.iter().sum::<f32>().max(1e-8);
    for c in 0..NUM_COMBOS {
        features.push(ranges[0].weights[c] / total_oop);
    }
    for c in 0..NUM_COMBOS {
        features.push(ranges[1].weights[c] / total_ip);
    }

    // Pot (normalized: divide by 2000 — covers pots up to ~2000 chips)
    features.push(pot as f32 / 2000.0);

    // Effective stack (normalized: divide by 2000)
    let eff_stack = stacks[0].min(stacks[1]);
    features.push(eff_stack as f32 / 2000.0);

    // Street: one-hot [turn, river]
    features.push(if street == Street::Turn { 1.0 } else { 0.0 });
    features.push(if street == Street::River { 1.0 } else { 0.0 });

    // Total reach per player (log-scaled, normalized by 8)
    features.push((total_oop + 1.0).ln() / 8.0);
    features.push((total_ip + 1.0).ln() / 8.0);

    debug_assert_eq!(features.len(), INPUT_DIM);
    features
}

/// Decode output tensor into per-combo EVs.
/// The model outputs 2652 floats: first 1326 = OOP EV, next 1326 = IP EV.
pub fn decode_output(output: &[f32], traverser: Player) -> [f32; NUM_COMBOS] {
    let mut ev = [0.0f32; NUM_COMBOS];
    let offset = if traverser == 0 { 0 } else { NUM_COMBOS };
    for c in 0..NUM_COMBOS {
        ev[c] = output[offset + c];
    }
    ev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_input_dimensions() {
        let board = Hand::new().add(0).add(1).add(2).add(3).add(4);
        let ranges = [Range::uniform(), Range::uniform()];
        let features = encode_input(board, &ranges, 100, [200, 200], Street::River);
        assert_eq!(features.len(), INPUT_DIM);
    }

    #[test]
    fn test_encode_input_board_encoding() {
        let board = Hand::new().add(0).add(10).add(20);
        let ranges = [Range::empty(), Range::empty()];
        let features = encode_input(board, &ranges, 50, [100, 100], Street::Turn);
        assert_eq!(features[0], 1.0);   // card 0 on board
        assert_eq!(features[1], 0.0);   // card 1 not on board
        assert_eq!(features[10], 1.0);  // card 10 on board
        assert_eq!(features[20], 1.0);  // card 20 on board
    }
}
