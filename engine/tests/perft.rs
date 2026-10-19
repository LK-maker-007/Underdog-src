use lk_47::board::Position;
use lk_47::movegen::perft;

// Leaf counts from the Chess Programming Wiki's Perft Results page, positions 1 to 6 and the mirror of 4.
const SUITE: [(&str, &[u64]); 7] = [
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[20, 400, 8_902, 197_281, 4_865_609, 119_060_324],
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        &[48, 2_039, 97_862, 4_085_603, 193_690_690],
    ),
    (
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[14, 191, 2_812, 43_238, 674_624, 11_030_083, 178_633_661],
    ),
    (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[6, 264, 9_467, 422_333, 15_833_292, 706_045_033],
    ),
    (
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        &[6, 264, 9_467, 422_333, 15_833_292, 706_045_033],
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[44, 1_486, 62_379, 2_103_487, 89_941_194],
    ),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        &[46, 2_079, 89_890, 3_894_594, 164_075_551],
    ),
];

fn check(limit: u64) {
    for (fen, counts) in SUITE {
        let pos = Position::from_fen(fen).unwrap();
        for (depth, &expected) in (1..).zip(counts.iter()).take_while(|&(_, &n)| n <= limit) {
            assert_eq!(perft(&pos, depth), expected, "{fen} at depth {depth}");
        }
    }
}

// The two positions with 218 legal moves given on the wiki's Encoding Moves page.
#[test]
fn most_moves_in_one_position() {
    for fen in [
        "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",
        "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1",
    ] {
        assert_eq!(perft(&Position::from_fen(fen).unwrap(), 1), 218, "{fen}");
    }
}

#[test]
fn perft_shallow() {
    check(1_000_000);
}

#[test]
#[ignore = "deep counts; run with cargo test --release -- --ignored"]
fn perft_full() {
    check(u64::MAX);
}
