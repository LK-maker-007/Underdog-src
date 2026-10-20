use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{env, io};

use lk_47::board::{Position, STARTPOS};
use lk_47::datagen::{Rng, play_game};
use lk_47::movegen::perft;
use lk_47::nnue::{Accumulator, Network};
use lk_47::search::{Limits, Searcher};
use lk_47::uci;

const USAGE: &str = "usage: lk-47 (UCI on stdin) | perft <depth> [fen] | bench [depth [network]] | openings <count> <seed> | datagen <games> <seed> <nodes> <file> | eval <network> <fen>";

// The perft suite positions, searched to a fixed depth so the node count is a signature of the search.
const BENCH: [&str; 7] = [
    STARTPOS,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
];

const OPENING_PLIES: usize = 8;
const OPENING_DEPTH: i32 = 4;
const OPENING_MAX_CP: i32 = 50;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match words.as_slice() {
        [] => uci::run(io::stdin().lock(), Arc::new(Mutex::new(io::stdout())))
            .map_err(|err| err.to_string()),
        ["perft", depth, fen @ ..] => {
            number(depth).and_then(|depth| perft_divide(depth as u32, fen))
        }
        ["bench"] => bench(5, None),
        ["bench", depth] => number(depth).and_then(|depth| bench(depth as i32, None)),
        ["bench", depth, net] => number(depth).and_then(|depth| bench(depth as i32, Some(net))),
        ["openings", count, seed] => {
            number(count).and_then(|count| openings(count as usize, number(seed)?))
        }
        ["datagen", games, seed, nodes, file] => number(games)
            .and_then(|games| datagen(games as usize, number(seed)?, number(nodes)?, file)),
        ["eval", net, fen @ ..] => static_eval(net, fen),
        _ => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn number(text: &str) -> Result<u64, String> {
    match text.parse::<u64>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("Expected a positive integer, got {text:?}")),
    }
}

fn perft_divide(depth: u32, fen: &[&str]) -> Result<(), String> {
    let pos = Position::from_fen(&if fen.is_empty() {
        STARTPOS.to_string()
    } else {
        fen.join(" ")
    })?;
    let start = Instant::now();
    let mut total = 0;
    for &m in pos.legal_moves().iter() {
        let nodes = perft(&pos.make_move(m), depth - 1);
        println!("{m}: {nodes}");
        total += nodes;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "\nnodes {total}\ntime {seconds:.3} s\nnps {:.0}",
        total as f64 / seconds
    );
    Ok(())
}

fn bench(depth: i32, net: Option<&str>) -> Result<(), String> {
    let net = net.map(read_network).transpose()?.map(Arc::new);
    let mut nodes = 0;
    let mut seconds = 0.0;
    for fen in BENCH {
        let mut searcher = Searcher::default();
        searcher.set_network(net.clone());
        let limits = Limits {
            depth,
            ..Limits::default()
        };
        // Only the search is timed, not setting up the table.
        let start = Instant::now();
        searcher.search(
            &Position::from_fen(fen)?,
            &[],
            limits,
            Arc::new(AtomicBool::new(false)),
            &mut |_| {},
        );
        seconds += start.elapsed().as_secs_f64();
        nodes += searcher.nodes();
    }
    println!(
        "nodes {nodes}\ntime {seconds:.3} s\nnps {:.0}",
        nodes as f64 / seconds
    );
    Ok(())
}

// Random play from the start position, kept only when a shallow search calls it close to level.
fn openings(count: usize, seed: u64) -> Result<(), String> {
    let mut rng = seed;
    let mut seen = HashSet::new();
    let mut written = 0;
    for _ in 0..count * 1000 {
        if written == count {
            return Ok(());
        }
        let mut pos = Position::from_fen(STARTPOS)?;
        for _ in 0..OPENING_PLIES {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            pos = pos.make_move(moves[(rng % moves.len() as u64) as usize]);
        }
        if pos.legal_moves().is_empty() || !seen.insert(pos.hash()) {
            continue;
        }
        let limits = Limits {
            depth: OPENING_DEPTH,
            ..Limits::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let found = Searcher::default().search(&pos, &[], limits, stop, &mut |_| {});
        if found.is_some_and(|(_, score)| score.abs() <= OPENING_MAX_CP) {
            println!("{}", pos.to_fen());
            written += 1;
        }
    }
    if written == count {
        Ok(())
    } else {
        Err(format!(
            "Found {written} of {count} openings within {OPENING_MAX_CP} cp"
        ))
    }
}

// Progress goes to stderr every 50 games, so a long run can be watched and the file holds only data.
fn datagen(games: usize, seed: u64, nodes: u64, path: &str) -> Result<(), String> {
    let file = File::create(path).map_err(|err| format!("Cannot create {path:?}: {err}"))?;
    let mut out = BufWriter::new(file);
    let mut rng = Rng::new(seed);
    let start = Instant::now();
    let mut positions = 0;
    for game in 1..=games {
        positions += play_game(&mut rng, nodes, &mut out).map_err(|err| err.to_string())?;
        if game % 50 == 0 || game == games {
            out.flush().map_err(|err| err.to_string())?;
            let seconds = start.elapsed().as_secs_f64();
            eprintln!(
                "games {game} positions {positions} seconds {seconds:.0} positions/s {:.0}",
                positions as f64 / seconds
            );
        }
    }
    Ok(())
}

// The network's static score for one position, from the side to move's point of view.
fn static_eval(net: &str, fen: &[&str]) -> Result<(), String> {
    let net = read_network(net)?;
    let pos = Position::from_fen(&fen.join(" "))?;
    println!("{}", net.evaluate(&Accumulator::new(&net, &pos), &pos));
    Ok(())
}

fn read_network(path: &str) -> Result<Network, String> {
    let bytes =
        std::fs::read(path).map_err(|err| format!("Cannot read network {path:?}: {err}"))?;
    Network::from_bytes(&bytes)
}
