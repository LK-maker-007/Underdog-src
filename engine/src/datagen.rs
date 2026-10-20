use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::board::{BISHOP, KNIGHT, PAWN, Position, QUEEN, ROOK, STARTPOS, WHITE};
use crate::search::{Limits, MATE_BOUND, Searcher};

const OPENING_PLIES: usize = 8;
// First guesses for ending a decided game early: a side scored at least WIN_SCORE for WIN_PLIES plies in a row,
// or both stayed within DRAW_SCORE for DRAW_PLIES plies once DRAW_FROM_PLY plies were played.
const WIN_SCORE: i32 = 2500;
const WIN_PLIES: usize = 4;
const DRAW_FROM_PLY: usize = 80;
const DRAW_SCORE: i32 = 10;
const DRAW_PLIES: usize = 10;
const MAX_PLIES: usize = 400;

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }

    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

// One self-play game. Writes a `<FEN> | <score> | <result>` line for each quiet position, score in centipawns and
// result as 1.0, 0.5 or 0.0, both from White's side. Returns the number of lines written.
pub fn play_game(rng: &mut Rng, nodes: u64, out: &mut impl Write) -> io::Result<usize> {
    let mut pos = Position::from_fen(STARTPOS).expect("the start position parses");
    for _ in 0..OPENING_PLIES {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return Ok(0);
        }
        pos = pos.make_move(moves[rng.below(moves.len())]);
    }

    let mut searcher = Searcher::default();
    let stop = Arc::new(AtomicBool::new(false));
    let mut history = Vec::new();
    let mut records = Vec::new();
    let (mut winning, mut level) = (0, 0);
    let result = loop {
        if pos.legal_moves().is_empty() {
            break match (pos.in_check(), pos.side) {
                (false, _) => 0.5,
                (true, WHITE) => 0.0,
                (true, _) => 1.0,
            };
        }
        if pos.halfmove >= 100
            || repetitions(&history, &pos) >= 2
            || insufficient(&pos)
            || history.len() >= MAX_PLIES
        {
            break 0.5;
        }
        let limits = Limits {
            nodes,
            ..Limits::default()
        };
        let (m, score) = searcher
            .search(&pos, &history, limits, Arc::clone(&stop), &mut |_| {})
            .expect("a position with legal moves has a best move");
        let white = if pos.side == WHITE { score } else { -score };

        winning = if white.abs() >= WIN_SCORE {
            winning + 1
        } else {
            0
        };
        if winning >= WIN_PLIES {
            break if white > 0 { 1.0 } else { 0.0 };
        }
        level = if history.len() >= DRAW_FROM_PLY && white.abs() <= DRAW_SCORE {
            level + 1
        } else {
            0
        };
        if level >= DRAW_PLIES {
            break 0.5;
        }

        if !pos.in_check() && !m.is_capture() && m.promotion().is_none() && score.abs() < MATE_BOUND
        {
            records.push((pos.to_fen(), white));
        }
        history.push(pos.hash());
        pos = pos.make_move(m);
    };

    for (fen, score) in &records {
        writeln!(out, "{fen} | {score} | {result:.1}")?;
    }
    Ok(records.len())
}

// Earlier occurrences of this position with the same side to move since the last capture or pawn move.
fn repetitions(history: &[u64], pos: &Position) -> usize {
    let reach = (pos.halfmove as usize).min(history.len());
    (2..=reach)
        .step_by(2)
        .filter(|&back| history[history.len() - back] == pos.hash())
        .count()
}

// Bare kings, or one minor piece against a bare king.
fn insufficient(pos: &Position) -> bool {
    (pos.pieces[PAWN] | pos.pieces[ROOK] | pos.pieces[QUEEN]) == 0
        && (pos.pieces[KNIGHT] | pos.pieces[BISHOP]).count_ones() <= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(seed: u64) -> String {
        let mut out = Vec::new();
        play_game(&mut Rng::new(seed), 400, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn lines_follow_the_documented_format() {
        let text = game(7);
        let lines: Vec<&str> = text.lines().collect();
        assert!(!lines.is_empty());
        let mut results = Vec::new();
        for line in &lines {
            let fields: Vec<&str> = line.split(" | ").collect();
            assert_eq!(fields.len(), 3, "{line}");
            let pos = Position::from_fen(fields[0]).unwrap();
            assert!(!pos.in_check(), "{line}");
            assert!(
                fields[1].parse::<i32>().unwrap().abs() < MATE_BOUND,
                "{line}"
            );
            assert!(["1.0", "0.5", "0.0"].contains(&fields[2]), "{line}");
            results.push(fields[2]);
        }
        results.dedup();
        assert_eq!(
            results.len(),
            1,
            "every line of one game carries that game's result"
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_game() {
        assert_eq!(game(11), game(11));
        assert_ne!(game(11), game(12));
    }

    #[test]
    fn repetitions_and_bare_material_are_seen() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 8 30").unwrap();
        assert_eq!(repetitions(&[pos.hash(), 1, pos.hash(), 2], &pos), 2);
        assert_eq!(repetitions(&[pos.hash(), 1, 2, 3], &pos), 1);
        assert!(!insufficient(&pos));
        assert!(insufficient(
            &Position::from_fen("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1").unwrap()
        ));
        assert!(!insufficient(
            &Position::from_fen("4k3/8/8/8/8/8/8/1NB1K3 w - - 0 1").unwrap()
        ));
    }
}
