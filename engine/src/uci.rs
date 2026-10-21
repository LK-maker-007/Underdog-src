use std::fs;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::board::{Position, STARTPOS, WHITE};
use crate::nnue::Network;
use crate::search::{HASH_MB, Limits, MAX_PLY, Searcher};

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

// Kept back from every move for process and pipe latency.
const MOVE_OVERHEAD_MS: i64 = 20;
const MAX_HASH_MB: usize = 65_536;
const MAX_THREADS: usize = 256;

pub type Output = Arc<Mutex<dyn Write + Send>>;

fn say(out: &Output, text: &str) -> io::Result<()> {
    let mut out = out.lock().unwrap();
    writeln!(out, "{text}")?;
    out.flush()
}

pub fn run(input: impl BufRead, out: Output) -> io::Result<()> {
    let mut pos = Position::from_fen(STARTPOS).expect("the start position parses");
    let mut history = Vec::new();
    let stop = Arc::new(AtomicBool::new(false));
    let mut net = load_network("").unwrap_or_else(|err| {
        let _ = say(&out, &format!("info string built-in network: {err}"));
        None
    });
    let (mut hash, mut threads) = (HASH_MB, 1);
    let mut idle = Some(Searcher::new(hash, threads, net.clone()));
    let mut running: Option<JoinHandle<Searcher>> = None;

    for line in input.lines() {
        let line = line?;
        let words: Vec<&str> = line.split_whitespace().collect();
        // The protocol skips unknown tokens and parses the rest of the line.
        let Some(at) = words.iter().position(|w| COMMANDS.contains(w)) else {
            continue;
        };
        let args = &words[at + 1..];
        match words[at] {
            "uci" => say(
                &out,
                &format!(
                    "id name LK-47 1.1\nid author Singaraj B\n\
                     option name EvalFile type string default <empty>\n\
                     option name Hash type spin default {HASH_MB} min 1 max {MAX_HASH_MB}\n\
                     option name Threads type spin default 1 min 1 max {MAX_THREADS}\nuciok"
                ),
            )?,
            "isready" => say(&out, "readyok")?,
            "ucinewgame" => {
                finish(&mut running, &mut idle, &stop);
                idle = Some(Searcher::new(hash, threads, net.clone()));
            }
            "setoption" => {
                finish(&mut running, &mut idle, &stop);
                if let Some(path) = option_value(args, "EvalFile") {
                    match load_network(&path) {
                        Ok(loaded) => {
                            net = loaded;
                            say(&out, &format!("info string network {path:?} loaded"))?;
                        }
                        Err(err) => say(&out, &format!("info string {err}"))?,
                    }
                    if let Some(searcher) = idle.as_mut() {
                        searcher.set_network(net.clone());
                    }
                }
                let mut resized = false;
                for (id, value, max) in [
                    ("Hash", &mut hash, MAX_HASH_MB),
                    ("Threads", &mut threads, MAX_THREADS),
                ] {
                    if let Some(text) = option_value(args, id) {
                        match text.parse::<usize>() {
                            Ok(n) if (1..=max).contains(&n) => {
                                *value = n;
                                resized = true;
                            }
                            _ => say(&out, &format!("info string {id} must be 1 to {max}"))?,
                        }
                    }
                }
                if resized {
                    idle = Some(Searcher::new(hash, threads, net.clone()));
                }
            }
            "position" => {
                finish(&mut running, &mut idle, &stop);
                match parse_position(args) {
                    Ok((next, seen)) => (pos, history) = (next, seen),
                    Err(err) => say(&out, &format!("info string {err}"))?,
                }
            }
            "go" => {
                finish(&mut running, &mut idle, &stop);
                let limits = parse_go(args, pos.side);
                let mut searcher = idle.take().expect("no search is running");
                stop.store(false, Ordering::Relaxed);
                let (stop, out, history) = (Arc::clone(&stop), Arc::clone(&out), history.clone());
                running = Some(thread::spawn(move || {
                    // A write fails only once the GUI has closed the pipe, and then nobody is listening.
                    let best = searcher.search(&pos, &history, limits, stop, &mut |line| {
                        let _ = say(&out, line);
                    });
                    let best = best.map_or("0000".to_string(), |(m, _)| m.to_string());
                    let _ = say(&out, &format!("bestmove {best}"));
                    searcher
                }));
            }
            "stop" => finish(&mut running, &mut idle, &stop),
            "quit" => break,
            _ => {}
        }
    }
    finish(&mut running, &mut idle, &stop);
    Ok(())
}

// `setoption name <id> value <x>`: the value of the named option, if this command sets it.
fn option_value(args: &[&str], id: &str) -> Option<String> {
    let name = args.iter().position(|&w| w == "name")?;
    let value = args
        .iter()
        .position(|&w| w == "value")
        .unwrap_or(args.len());
    let given = args.get(name + 1..value)?.join(" ");
    given
        .eq_ignore_ascii_case(id)
        .then(|| args.get(value + 1..).unwrap_or(&[]).join(" "))
}

// An empty path means the built-in network, or the hand-written evaluation in a binary built without one.
fn load_network(path: &str) -> Result<Option<Arc<Network>>, String> {
    if path.is_empty() || path == "<empty>" {
        return Ok(Network::built_in()?.map(Arc::new));
    }
    let bytes = fs::read(path).map_err(|err| format!("Cannot read network {path:?}: {err}"))?;
    Ok(Some(Arc::new(Network::from_bytes(&bytes)?)))
}

fn finish(
    running: &mut Option<JoinHandle<Searcher>>,
    idle: &mut Option<Searcher>,
    stop: &AtomicBool,
) {
    if let Some(handle) = running.take() {
        stop.store(true, Ordering::Relaxed);
        *idle = Some(handle.join().expect("the search thread panicked"));
    }
}

fn parse_go(args: &[&str], side: usize) -> Limits {
    let value = |name: &str| {
        let at = args.iter().position(|&w| w == name)?;
        args.get(at + 1)?.parse::<i64>().ok()
    };
    let ms = |ms: i64| Duration::from_millis(ms.max(0) as u64);
    let mut limits = Limits::default();
    if let Some(depth) = value("depth") {
        limits.depth = depth.clamp(1, MAX_PLY as i64 - 1) as i32;
    }
    if let Some(nodes) = value("nodes") {
        limits.nodes = nodes.max(1) as u64;
    }
    if let Some(movetime) = value("movetime") {
        limits.soft = Some(ms(movetime - MOVE_OVERHEAD_MS));
        limits.hard = Some(ms(movetime - MOVE_OVERHEAD_MS));
    }
    let (time, increment) = if side == WHITE {
        ("wtime", "winc")
    } else {
        ("btime", "binc")
    };
    if let Some(left) = value(time) {
        let left = (left - MOVE_OVERHEAD_MS).max(0);
        let increment = value(increment).unwrap_or(0).max(0);
        let moves_to_go = value("movestogo").unwrap_or(20).max(1);
        let hard = left / 2;
        limits.soft = Some(ms((left / moves_to_go + increment * 3 / 4).min(hard)));
        limits.hard = Some(ms(hard));
    }
    limits
}

fn parse_position(words: &[&str]) -> Result<(Position, Vec<u64>), String> {
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
    let mut history = Vec::new();
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
                history.push(pos.hash);
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
    Ok((pos, history))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn talk(input: &str) -> Vec<String> {
        let out = Arc::new(Mutex::new(Vec::new()));
        run(input.as_bytes(), out.clone()).unwrap();
        let bytes = out.lock().unwrap().clone();
        String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn legal_after(words: &[&str], best: &str) -> bool {
        let (pos, _) = parse_position(words).unwrap();
        pos.legal_moves().iter().any(|m| m.to_string() == best)
    }

    #[test]
    fn handshake() {
        assert_eq!(
            talk("uci\nisready\n"),
            [
                "id name LK-47 1.1",
                "id author Singaraj B",
                "option name EvalFile type string default <empty>",
                "option name Hash type spin default 16 min 1 max 65536",
                "option name Threads type spin default 1 min 1 max 256",
                "uciok",
                "readyok"
            ]
        );
    }

    #[test]
    fn hash_and_threads_are_set_and_checked() {
        let lines = talk(
            "setoption name Hash value 0\nsetoption name Threads value many\n\
             setoption name Hash value 8\nsetoption name Threads value 3\n\
             position startpos\ngo depth 4\nisready\n",
        );
        assert_eq!(lines[0], "info string Hash must be 1 to 65536");
        assert_eq!(lines[1], "info string Threads must be 1 to 256");
        let best = lines
            .iter()
            .find_map(|l| l.strip_prefix("bestmove "))
            .unwrap();
        assert!(legal_after(&["startpos"], best), "{lines:?}");
    }

    #[test]
    fn eval_file_is_read_from_setoption() {
        let words = |text: &'static str| text.split(' ').collect::<Vec<_>>();
        assert_eq!(
            option_value(&words("name EvalFile value /tmp/a b.bin"), "EvalFile"),
            Some("/tmp/a b.bin".to_string())
        );
        assert_eq!(
            option_value(&words("name evalfile value x"), "EvalFile"),
            Some("x".to_string())
        );
        assert_eq!(option_value(&words("name Hash value 64"), "EvalFile"), None);
        let lines = talk("setoption name EvalFile value /nonexistent/net.bin\nisready\n");
        assert!(
            lines[0].starts_with("info string Cannot read network"),
            "{lines:?}"
        );
        assert_eq!(lines[1], "readyok");
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
        let (pos, history) = parse_position(&words.split(' ').collect::<Vec<_>>()).unwrap();
        assert_eq!(pos.to_fen(), "1Q6/4k3/8/8/8/8/8/R4RK1 b - - 2 2");
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn go_reports_and_answers_a_legal_move() {
        let lines = talk("position startpos moves e2e4 e7e5 g1f3\ngo depth 2\n");
        let (best, info) = lines.split_last().unwrap();
        assert!(
            info.iter().all(|line| line.starts_with("info depth ")),
            "{lines:?}"
        );
        let best = best.strip_prefix("bestmove ").unwrap();
        assert!(
            legal_after(&["startpos", "moves", "e2e4", "e7e5", "g1f3"], best),
            "{best}"
        );
    }

    #[test]
    fn bad_position_commands_are_reported_and_ignored() {
        let lines = talk(concat!(
            "position startpos moves e2e4\n",
            "position startpos moves e7e5\n",
            "position fen 8/8/8/8/8/8/8/8 w - - 0 1\n",
            "position\n",
            "position startpos e2e4\n",
            "go depth 1\n",
        ));
        assert!(
            lines[0].starts_with("info string Illegal move \"e7e5\""),
            "{}",
            lines[0]
        );
        assert!(lines[1].contains("one white king"), "{}", lines[1]);
        assert!(lines[2].contains("needs startpos or fen"), "{}", lines[2]);
        assert!(lines[3].contains("Expected moves"), "{}", lines[3]);
        let best = lines.last().unwrap().strip_prefix("bestmove ").unwrap();
        assert!(legal_after(&["startpos", "moves", "e2e4"], best), "{best}");
    }

    #[test]
    fn a_mated_side_answers_the_null_move() {
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
        assert_eq!(
            talk(&format!("position fen {fen}\ngo\n")),
            ["bestmove 0000"]
        );
    }

    #[test]
    fn clock_limits_keep_a_reserve() {
        let limits = parse_go(
            &["wtime", "8000", "btime", "100", "winc", "80", "binc", "80"],
            WHITE,
        );
        assert_eq!(limits.soft, Some(Duration::from_millis(7980 / 20 + 60)));
        assert_eq!(limits.hard, Some(Duration::from_millis(7980 / 2)));
        let limits = parse_go(&["wtime", "8000", "btime", "-50"], crate::board::BLACK);
        assert_eq!(limits.hard, Some(Duration::ZERO));
        let limits = parse_go(&["movetime", "500", "depth", "999"], WHITE);
        assert_eq!(limits.soft, Some(Duration::from_millis(480)));
        assert_eq!(limits.depth, MAX_PLY as i32 - 1);
    }
}
