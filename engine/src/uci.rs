use std::io::{self, BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::board::{Position, STARTPOS};

const COMMANDS: [&str; 11] = [
    "uci",
    "debug",
    "isready",
    "setoption",
    "register",
    "ucinewgame",
    "position",
    "go",
    "stop",
    "ponderhit",
    "quit",
];

pub fn run(input: impl BufRead, out: &mut impl Write) -> io::Result<()> {
    let mut pos = Position::from_fen(STARTPOS).expect("the start position parses");
    let mut rng = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64)
        | 1;
    for line in input.lines() {
        let line = line?;
        let words: Vec<&str> = line.split_whitespace().collect();
        // The protocol skips unknown tokens and parses the rest of the line.
        let Some(at) = words.iter().position(|w| COMMANDS.contains(w)) else {
            continue;
        };
        match words[at] {
            "uci" => writeln!(out, "id name LK-47\nid author Singaraj B\nuciok")?,
            "isready" => writeln!(out, "readyok")?,
            "position" => match parse_position(&words[at + 1..]) {
                Ok(next) => pos = next,
                Err(err) => writeln!(out, "info string {err}")?,
            },
            "go" => {
                // Until the search exists, go plays a random legal move. It evaluates nothing, so it
                // reports depth 0 and score 0 only because match runners expect a scored info line.
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    writeln!(out, "bestmove 0000")?;
                } else {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let best = moves[(rng % moves.len() as u64) as usize];
                    writeln!(out, "info depth 0 score cp 0 pv {best}\nbestmove {best}")?;
                }
            }
            "quit" => break,
            _ => {}
        }
        out.flush()?;
    }
    Ok(())
}

fn parse_position(words: &[&str]) -> Result<Position, String> {
    let (mut pos, rest) = match words {
        ["startpos", rest @ ..] => (Position::from_fen(STARTPOS)?, rest),
        ["fen", rest @ ..] => {
            let end = rest
                .iter()
                .position(|&w| w == "moves")
                .unwrap_or(rest.len());
            (Position::from_fen(&rest[..end].join(" "))?, &rest[end..])
        }
        _ => {
            return Err(format!(
                "position needs startpos or fen, got {:?}",
                words.join(" ")
            ));
        }
    };
    match rest {
        [] => {}
        ["moves", moves @ ..] => {
            for text in moves {
                let m = pos
                    .legal_moves()
                    .iter()
                    .copied()
                    .find(|m| m.to_string() == *text)
                    .ok_or_else(|| format!("Illegal move {text:?} in {}", pos.to_fen()))?;
                pos = pos.make_move(m);
            }
        }
        other => {
            return Err(format!(
                "Expected moves after the position, got {:?}",
                other.join(" ")
            ));
        }
    }
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn talk(input: &str) -> Vec<String> {
        let mut out = Vec::new();
        run(input.as_bytes(), &mut out).unwrap();
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn handshake() {
        assert_eq!(
            talk("uci\nisready\n"),
            ["id name LK-47", "id author Singaraj B", "uciok", "readyok"]
        );
    }

    #[test]
    fn unknown_tokens_before_a_command_are_skipped() {
        assert_eq!(
            talk("joho isready\nnonsense\nsetoption name Style value go\n"),
            ["readyok"]
        );
    }

    #[test]
    fn moves_are_read_in_uci_notation() {
        let words = "fen 4k3/1P6/8/8/8/8/8/R3K2R w KQ - 0 1 moves b7b8q e8e7 e1g1";
        let pos = parse_position(&words.split(' ').collect::<Vec<_>>()).unwrap();
        assert_eq!(pos.to_fen(), "1Q6/4k3/8/8/8/8/8/R4RK1 b - - 2 2");
    }

    #[test]
    fn go_answers_a_legal_move() {
        let pos = parse_position(&["startpos", "moves", "e2e4", "e7e5", "g1f3"]).unwrap();
        let legal: Vec<String> = pos.legal_moves().iter().map(|m| m.to_string()).collect();
        for _ in 0..20 {
            let lines = talk("position startpos moves e2e4 e7e5 g1f3\ngo wtime 1000 btime 1000\n");
            assert_eq!(lines.len(), 2);
            let best = lines[1].strip_prefix("bestmove ").unwrap();
            assert_eq!(lines[0], format!("info depth 0 score cp 0 pv {best}"));
            assert!(legal.iter().any(|m| m == best), "{best} is not legal");
        }
    }

    #[test]
    fn bad_position_commands_are_reported_and_ignored() {
        let lines = talk(concat!(
            "position startpos moves e2e4\n",
            "position startpos moves e7e5\n",
            "position fen 8/8/8/8/8/8/8/8 w - - 0 1\n",
            "position\n",
            "position startpos e2e4\n",
            "go\n",
        ));
        assert!(
            lines[0].starts_with("info string Illegal move \"e7e5\""),
            "{}",
            lines[0]
        );
        assert!(lines[1].contains("one white king"), "{}", lines[1]);
        assert!(lines[2].contains("needs startpos or fen"), "{}", lines[2]);
        assert!(lines[3].contains("Expected moves"), "{}", lines[3]);
        let after_e4 = parse_position(&["startpos", "moves", "e2e4"]).unwrap();
        let best = lines[5].strip_prefix("bestmove ").unwrap();
        assert!(
            after_e4.legal_moves().iter().any(|m| m.to_string() == best),
            "{best}"
        );
        assert_eq!(lines.len(), 6);
    }

    #[test]
    fn a_mated_side_answers_the_null_move() {
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
        assert_eq!(
            talk(&format!("position fen {fen}\ngo\n")),
            ["bestmove 0000"]
        );
    }
}
