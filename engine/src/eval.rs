use crate::board::{BISHOP, BLACK, KING, KNIGHT, PAWN, Position, QUEEN, ROOK, WHITE, squares};

// Middlegame and endgame values in centipawns. These are first guesses, made only to generate the first training
// data, and are replaced by the network.
const MATERIAL: [(i32, i32); 6] = [
    (100, 120),
    (320, 300),
    (330, 320),
    (500, 550),
    (950, 1000),
    (0, 0),
];
const PHASE: [i32; 6] = [0, 1, 1, 2, 4, 0];
const FULL_PHASE: i32 = 24;

// Square bonuses from White's side, built from simple rules: knights, bishops and queens prefer the centre, pawns
// gain as they advance, rooks like the seventh rank, and the king hides early and centralises late.
static SQUARES: [[(i32, i32); 64]; 6] = {
    let mut table = [[(0, 0); 64]; 6];
    let mut sq = 0;
    while sq < 64 {
        let (file, rank) = ((sq % 8) as i32, (sq / 8) as i32);
        let near_file = if file < 7 - file { file } else { 7 - file };
        let near_rank = if rank < 7 - rank { rank } else { 7 - rank };
        let centre = near_file + near_rank;
        let central_pawn = if (file == 3 || file == 4) && (rank == 3 || rank == 4) {
            10
        } else {
            0
        };
        table[PAWN][sq] = (5 * (rank - 1) + central_pawn, 10 * (rank - 1));
        table[KNIGHT][sq] = (5 * centre - 15, 4 * centre - 12);
        table[BISHOP][sq] = (3 * centre - 9, 3 * centre - 9);
        let seventh = if rank == 6 { 1 } else { 0 };
        let central_file = if file == 3 || file == 4 { 5 } else { 0 };
        table[ROOK][sq] = (15 * seventh + central_file, 10 * seventh);
        table[QUEEN][sq] = (centre - 3, 4 * centre - 12);
        let shelter = [10, 15, 5, 0, 0, 5, 15, 10];
        let king_mg = if rank == 0 {
            shelter[file as usize]
        } else {
            -15 * rank
        };
        table[KING][sq] = (king_mg, 6 * centre - 18);
        sq += 1;
    }
    table
};

// From the side to move's point of view.
pub fn evaluate(pos: &Position) -> i32 {
    let (mut mg, mut eg, mut phase) = (0, 0, 0);
    for color in [WHITE, BLACK] {
        let sign = if color == WHITE { 1 } else { -1 };
        for kind in PAWN..=KING {
            for sq in squares(pos.pieces[kind] & pos.colors[color]) {
                let from_white = if color == WHITE { sq } else { sq ^ 56 };
                mg += sign * (MATERIAL[kind].0 + SQUARES[kind][from_white].0);
                eg += sign * (MATERIAL[kind].1 + SQUARES[kind][from_white].1);
                phase += PHASE[kind];
            }
        }
    }
    let phase = phase.min(FULL_PHASE);
    let score = (mg * phase + eg * (FULL_PHASE - phase)) / FULL_PHASE;
    if pos.side == WHITE { score } else { -score }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTPOS;

    // The same position with colours swapped and the board turned over.
    fn mirror(fen: &str) -> String {
        let fields: Vec<&str> = fen.split(' ').collect();
        let swap_case = |c: char| {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else {
                c.to_ascii_uppercase()
            }
        };
        let board: Vec<String> = fields[0]
            .split('/')
            .rev()
            .map(|rank| rank.chars().map(swap_case).collect())
            .collect();
        let side = if fields[1] == "w" { "b" } else { "w" };
        let mut castling: Vec<char> = fields[2].chars().map(swap_case).collect();
        castling.sort_by_key(|c| "KQkq-".find(*c));
        let ep = match fields[3].as_bytes() {
            [file, rank] => format!("{}{}", *file as char, (b'9' - rank + b'0') as char),
            _ => "-".to_string(),
        };
        format!(
            "{} {side} {} {ep} {} {}",
            board.join("/"),
            castling.iter().collect::<String>(),
            fields[4],
            fields[5]
        )
    }

    #[test]
    fn mirrored_positions_score_the_same() {
        for fen in [
            STARTPOS,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let flipped = Position::from_fen(&mirror(fen)).unwrap();
            assert_eq!(
                evaluate(&pos),
                evaluate(&flipped),
                "{fen} against {}",
                mirror(fen)
            );
        }
    }

    #[test]
    fn the_start_is_level_and_a_spare_knight_is_not() {
        assert_eq!(evaluate(&Position::from_fen(STARTPOS).unwrap()), 0);
        let up_a_knight =
            Position::from_fen("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(evaluate(&up_a_knight) > 250, "{}", evaluate(&up_a_knight));
    }

    #[test]
    fn square_rules_point_the_right_way() {
        assert!(
            SQUARES[KNIGHT][27].0 > SQUARES[KNIGHT][0].0,
            "d4 beats a1 for a knight"
        );
        assert!(
            SQUARES[PAWN][48].1 > SQUARES[PAWN][8].1,
            "a pawn on the seventh beats one on the second"
        );
        assert!(
            SQUARES[KING][6].0 > SQUARES[KING][28].0,
            "g1 is safer than e4 in the middlegame"
        );
        assert!(
            SQUARES[KING][28].1 > SQUARES[KING][6].1,
            "e4 is better than g1 in the endgame"
        );
    }
}
