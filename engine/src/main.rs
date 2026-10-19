use std::process::ExitCode;
use std::time::Instant;
use std::{env, io};

use lk_47::board::{Position, STARTPOS};
use lk_47::movegen::perft;
use lk_47::uci;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return match uci::run(io::stdin().lock(), &mut io::stdout().lock()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        };
    }
    let [command, depth, fen @ ..] = args.as_slice() else {
        eprintln!("usage: lk-47 (UCI on stdin), or lk-47 perft <depth> [fen]");
        return ExitCode::FAILURE;
    };
    if command != "perft" {
        eprintln!("Unknown command {command:?}");
        return ExitCode::FAILURE;
    }
    let depth = match depth.parse::<u32>() {
        Ok(d) if d > 0 => d,
        _ => {
            eprintln!("Depth must be a positive integer, got {depth:?}");
            return ExitCode::FAILURE;
        }
    };
    let fen = if fen.is_empty() {
        STARTPOS.to_string()
    } else {
        fen.join(" ")
    };
    let pos = match Position::from_fen(&fen) {
        Ok(pos) => pos,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };

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
    ExitCode::SUCCESS
}
