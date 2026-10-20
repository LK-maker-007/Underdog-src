use crate::board::{BLACK, EMPTY, KING, PAWN, Position, ROOK, WHITE, squares};
use crate::movegen::Move;

// The network file's contract with `trainer/src/main.rs`: (768 -> HIDDEN)x2 -> 1 with SCReLU, the first layer
// quantised by QA, the output weights by QB, and the output scaled by SCALE into centipawns.
pub const HIDDEN: usize = 512;
const SCALE: i32 = 400;
const QA: i32 = 255;
const QB: i32 = 64;

const FEATURES: usize = 768;
// Keeps a clamped value times an output weight within i16: 255 * 128 = 32,640.
const MAX_OUTPUT_WEIGHT: u16 = 128;
// i16 values in file order: first-layer weights by feature, first-layer biases, output weights, output bias.
const VALUES: usize = FEATURES * HIDDEN + HIDDEN + 2 * HIDDEN + 1;

static BUILT_IN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net.bin"));

pub struct Network {
    weights: Vec<[i16; HIDDEN]>,
    bias: [i16; HIDDEN],
    output: [i16; 2 * HIDDEN],
    output_bias: i16,
}

impl Network {
    // bullet writes little-endian i16 values, the first layer column-major so each feature's weights are
    // contiguous, and pads the file to a multiple of 64 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        let padded = (VALUES * 2).div_ceil(64) * 64;
        if bytes.len() != padded {
            return Err(format!(
                "A network must be {padded} bytes, got {}",
                bytes.len()
            ));
        }
        let mut values = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i16::from_le_bytes(pair));
        let mut take = || values.next().expect("the length was checked");
        let weights = (0..FEATURES)
            .map(|_| std::array::from_fn(|_| take()))
            .collect();
        let bias = std::array::from_fn(|_| take());
        let output: [i16; 2 * HIDDEN] = std::array::from_fn(|_| take());
        let output_bias = take();
        if let Some(weight) = output.iter().find(|w| w.unsigned_abs() > MAX_OUTPUT_WEIGHT) {
            return Err(format!(
                "An output weight of {weight} is beyond the supported {MAX_OUTPUT_WEIGHT}"
            ));
        }
        Ok(Network {
            weights,
            bias,
            output,
            output_bias,
        })
    }

    // The network named by LK47_NET when the binary was built, if one was.
    pub fn built_in() -> Result<Option<Network>, String> {
        if BUILT_IN.is_empty() {
            Ok(None)
        } else {
            Network::from_bytes(BUILT_IN).map(Some)
        }
    }

    // Centipawns from the side to move's point of view; `acc` is the accumulator of `pos`.
    pub fn evaluate(&self, acc: &Accumulator, pos: &Position) -> i32 {
        let side = pos.side;
        let half = |values: &[i16; HIDDEN], weights: &[i16]| -> i32 {
            values
                .iter()
                .zip(weights)
                .map(|(&x, &w)| {
                    let v = x.clamp(0, QA as i16);
                    i32::from(v * w) * i32::from(v)
                })
                .sum()
        };
        let sum = half(&acc.values[side], &self.output[..HIDDEN])
            + half(&acc.values[side ^ 1], &self.output[HIDDEN..]);
        (sum / QA + i32::from(self.output_bias)) * SCALE / (QA * QB)
    }
}

// Each side's view of the board: its own pieces count first, and Black sees the board turned over.
fn feature(perspective: usize, piece: u8, sq: usize) -> usize {
    let (color, kind) = (piece as usize / 6, piece as usize % 6);
    let theirs = if color == perspective { 0 } else { 384 };
    let sq = if perspective == WHITE { sq } else { sq ^ 56 };
    theirs + 64 * kind + sq
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Accumulator {
    values: [[i16; HIDDEN]; 2],
}

impl Accumulator {
    pub fn new(net: &Network, pos: &Position) -> Accumulator {
        let mut acc = Accumulator {
            values: [net.bias; 2],
        };
        for sq in squares(pos.colors[WHITE] | pos.colors[BLACK]) {
            acc.change(net, pos.board[sq], sq, true);
        }
        acc
    }

    fn change(&mut self, net: &Network, piece: u8, sq: usize, add: bool) {
        for perspective in [WHITE, BLACK] {
            let weights = &net.weights[feature(perspective, piece, sq)];
            for (value, &weight) in self.values[perspective].iter_mut().zip(weights) {
                *value = if add {
                    value.wrapping_add(weight)
                } else {
                    value.wrapping_sub(weight)
                };
            }
        }
    }

    // Sets this to the accumulator of `_child`, the position after `m`, from `prev`, the accumulator of `pos`, in one
    // pass per perspective. The child is unused here; a network with king buckets needs it to rebuild a side.
    pub fn update(
        &mut self,
        prev: &Accumulator,
        net: &Network,
        pos: &Position,
        _child: &Position,
        m: Move,
    ) {
        let (from, to) = (m.from(), m.to());
        let moved = pos.board[from];
        let color = moved as usize / 6;
        let placed = m.promotion().map_or(moved, |kind| (color * 6 + kind) as u8);
        let rook = (color * 6 + ROOK) as u8;
        let taken = match m.flag() {
            Move::EN_PASSANT => Some((pos.board[to ^ 8], to ^ 8)),
            _ if m.is_capture() => Some((pos.board[to], to)),
            _ => None,
        };
        let castle = match m.flag() {
            Move::KING_CASTLE => Some((to + 1, to - 1)),
            Move::QUEEN_CASTLE => Some((to - 2, to + 1)),
            _ => None,
        };
        debug_assert!(moved != EMPTY && (moved as usize % 6 == PAWN || m.promotion().is_none()));
        debug_assert!(castle.is_none() || moved as usize % 6 == KING);
        for perspective in [WHITE, BLACK] {
            let row = |piece: u8, sq: usize| &net.weights[feature(perspective, piece, sq)];
            let (out, input) = (&mut self.values[perspective], &prev.values[perspective]);
            let (add, sub) = (row(placed, to), row(moved, from));
            match (taken, castle) {
                (Some((piece, sq)), _) => apply(out, input, [add], [sub, row(piece, sq)]),
                (None, Some((rook_from, rook_to))) => apply(
                    out,
                    input,
                    [add, row(rook, rook_to)],
                    [sub, row(rook, rook_from)],
                ),
                (None, None) => apply(out, input, [add], [sub]),
            }
        }
    }
}

fn apply<const A: usize, const S: usize>(
    out: &mut [i16; HIDDEN],
    input: &[i16; HIDDEN],
    adds: [&[i16; HIDDEN]; A],
    subs: [&[i16; HIDDEN]; S],
) {
    for i in 0..HIDDEN {
        let mut value = input[i];
        for add in adds {
            value = value.wrapping_add(add[i]);
        }
        for sub in subs {
            value = value.wrapping_sub(sub[i]);
        }
        out[i] = value;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::board::STARTPOS;

    // A network of small pseudo-random weights, laid out exactly as a file from bullet.
    pub(crate) fn random_network() -> Network {
        let mut state = 0x5DEE_CE66_D1A4_F87D_u64;
        let mut bytes = Vec::new();
        for _ in 0..VALUES {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.extend_from_slice(&((state % 129) as i16 - 64).to_le_bytes());
        }
        bytes.resize((VALUES * 2).div_ceil(64) * 64, 0);
        Network::from_bytes(&bytes).unwrap()
    }

    fn walk(net: &Network, pos: &Position, acc: &Accumulator, depth: u32) {
        assert_eq!(*acc, Accumulator::new(net, pos), "{}", pos.to_fen());
        if depth > 0 {
            for &m in pos.legal_moves().iter() {
                // A stale destination, so any value the update fails to write shows up.
                let mut next = Accumulator::new(net, &Position::from_fen(STARTPOS).unwrap());
                let child = pos.make_move(m);
                next.update(acc, net, pos, &child, m);
                walk(net, &child, &next, depth - 1);
            }
        }
    }

    #[test]
    fn updates_match_a_full_refresh() {
        let net = random_network();
        for fen in [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            walk(&net, &pos, &Accumulator::new(&net, &pos), 3);
        }
    }

    #[test]
    fn a_mirrored_position_scores_the_same() {
        let net = random_network();
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let flipped = Position::from_fen(
            "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1",
        )
        .unwrap();
        let score = net.evaluate(&Accumulator::new(&net, &pos), &pos);
        assert_eq!(
            score,
            net.evaluate(&Accumulator::new(&net, &flipped), &flipped)
        );
    }

    #[test]
    fn evaluation_matches_the_plain_formula() {
        let net = random_network();
        for fen in [
            STARTPOS,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let acc = Accumulator::new(&net, &pos);
            let side = pos.side;
            let mut sum = 0;
            for i in 0..HIDDEN {
                let ours = i32::from(acc.values[side][i]).clamp(0, QA);
                let theirs = i32::from(acc.values[side ^ 1][i]).clamp(0, QA);
                sum += ours * ours * i32::from(net.output[i])
                    + theirs * theirs * i32::from(net.output[HIDDEN + i]);
            }
            let plain = (sum / QA + i32::from(net.output_bias)) * SCALE / (QA * QB);
            assert_eq!(net.evaluate(&acc, &pos), plain, "{fen}");
        }
    }

    #[test]
    fn an_output_weight_too_large_is_refused() {
        let mut bytes = vec![0; (VALUES * 2).div_ceil(64) * 64];
        let at = 2 * (FEATURES * HIDDEN + HIDDEN + 7);
        bytes[at..at + 2].copy_from_slice(&129_i16.to_le_bytes());
        let err = Network::from_bytes(&bytes).err().unwrap();
        assert!(err.contains("129"), "{err}");
        bytes[at..at + 2].copy_from_slice(&(-128_i16).to_le_bytes());
        assert!(Network::from_bytes(&bytes).is_ok());
    }

    #[test]
    fn the_built_in_network_loads() {
        assert!(Network::built_in().is_ok());
    }

    #[test]
    fn a_file_of_the_wrong_size_is_refused() {
        let err = Network::from_bytes(&[0; 100]).err().unwrap();
        assert!(err.contains("must be"), "{err}");
    }
}
