use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::attacks::{bishop_attacks, rook_attacks};
use crate::board::{BISHOP, BLACK, EMPTY, KING, PAWN, Position, QUEEN, ROOK, WHITE, squares};
use crate::eval::evaluate;
use crate::movegen::Move;
use crate::nnue::{Accumulator, Network};
use crate::tt::{EXACT, LOWER, Table, UPPER};

pub const MAX_PLY: usize = 128;
pub const INFINITY: i32 = 32_000;
pub const MATE: i32 = 31_000;
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;
pub const HASH_MB: usize = 16;

// Move ordering bands, best first: table move, captures, queen promotions, killers, then quiet moves by history.
const TT_KEY: i32 = 2_000_000;
const CAPTURE_KEY: i32 = 1_000_000;
const PROMOTION_KEY: i32 = 900_000;
const KILLER_KEY: i32 = 800_000;

// First guesses, to be tuned by SPRT.
const ASPIRATION_WINDOW: i32 = 25;
const NULL_REDUCTION: i32 = 3;
const RFP_DEPTH: i32 = 6;
const RFP_MARGIN: i32 = 80;
const LMR_FULL_MOVES: usize = 3;
const LMR_BASE: f64 = 0.75;
const LMR_DIVISOR: f64 = 2.25;
const LMP_DEPTH: i32 = 8;
const FUTILITY_DEPTH: i32 = 6;
const FUTILITY_BASE: i32 = 100;
const FUTILITY_MARGIN: i32 = 90;
const HISTORY_MAX: i32 = 16_384;
const IIR_DEPTH: i32 = 4;
const SINGULAR_DEPTH: i32 = 8;
const SINGULAR_MARGIN: i32 = 2;

// Piece values for exchanges only; the king's is large so that taking a defended piece with it never pays.
const SEE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 20_000];

pub struct Limits {
    pub depth: i32,
    pub nodes: u64,
    pub soft: Option<Duration>,
    pub hard: Option<Duration>,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            depth: MAX_PLY as i32 - 1,
            nodes: u64::MAX,
            soft: None,
            hard: None,
        }
    }
}

pub struct Searcher {
    nodes: u64,
    stopped: bool,
    start: Instant,
    limits: Limits,
    stop: Arc<AtomicBool>,
    // Hashes of the game so far and of the current search path, for repetition checks.
    path: Vec<u64>,
    pv: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    tt: Arc<Table>,
    net: Option<Arc<Network>>,
    // Lazy SMP: searchers on other threads that share the table and stop when this one does.
    helpers: Vec<Searcher>,
    // One accumulator per ply, filled only when a network is loaded.
    accumulators: Vec<Accumulator>,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    // The move a singular extension search leaves out at each ply, or the null move.
    excluded: [Move; MAX_PLY],
    // Late move reductions by depth and move number.
    reductions: Box<[[i32; 64]; 64]>,
}

impl Default for Searcher {
    fn default() -> Searcher {
        Searcher::new(HASH_MB, 1, None)
    }
}

impl Searcher {
    pub fn new(megabytes: usize, threads: usize, net: Option<Arc<Network>>) -> Searcher {
        let tt = Arc::new(Table::new(megabytes));
        let mut main = Searcher::sharing(Arc::clone(&tt), net.clone());
        main.helpers = (1..threads)
            .map(|_| Searcher::sharing(Arc::clone(&tt), net.clone()))
            .collect();
        main
    }

    fn sharing(tt: Arc<Table>, net: Option<Arc<Network>>) -> Searcher {
        Searcher {
            nodes: 0,
            stopped: false,
            start: Instant::now(),
            limits: Limits::default(),
            stop: Arc::new(AtomicBool::new(false)),
            path: Vec::new(),
            pv: [[Move::default(); MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            tt,
            net,
            helpers: Vec::new(),
            accumulators: Vec::new(),
            killers: [[Move::default(); 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            excluded: [Move::default(); MAX_PLY],
            reductions: Box::new(std::array::from_fn(|depth| {
                std::array::from_fn(|moves| {
                    let (d, m) = (depth.max(1) as f64, moves.max(1) as f64);
                    (LMR_BASE + d.ln() * m.ln() / LMR_DIVISOR) as i32
                })
            })),
        }
    }

    pub fn set_network(&mut self, net: Option<Arc<Network>>) {
        for helper in &mut self.helpers {
            helper.net = net.clone();
        }
        self.net = net;
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    // Only this thread's search reports, counts toward the limits and picks the move.
    pub fn search(
        &mut self,
        pos: &Position,
        history: &[u64],
        limits: Limits,
        stop: Arc<AtomicBool>,
        report: &mut dyn FnMut(&str),
    ) -> Option<(Move, i32)> {
        let mut helpers = std::mem::take(&mut self.helpers);
        let done = Arc::new(AtomicBool::new(false));
        let best = thread::scope(|scope| {
            for helper in &mut helpers {
                let done = Arc::clone(&done);
                scope.spawn(move || {
                    helper.iterate(pos, history, Limits::default(), done, &mut |_| {})
                });
            }
            let best = self.iterate(pos, history, limits, stop, report);
            done.store(true, Ordering::Relaxed);
            best
        });
        self.helpers = helpers;
        best
    }

    fn iterate(
        &mut self,
        pos: &Position,
        history: &[u64],
        limits: Limits,
        stop: Arc<AtomicBool>,
        report: &mut dyn FnMut(&str),
    ) -> Option<(Move, i32)> {
        self.nodes = 0;
        self.stopped = false;
        self.start = Instant::now();
        self.limits = limits;
        self.stop = stop;
        self.path.clear();
        self.path.extend_from_slice(history);
        if let Some(net) = self.net.as_deref() {
            self.accumulators = vec![Accumulator::new(net, pos); MAX_PLY + 1];
        }
        self.killers = [[Move::default(); 2]; MAX_PLY];
        for score in self.history.iter_mut().flatten().flatten() {
            *score /= 2;
        }

        if pos.legal_moves().is_empty() {
            return None;
        }
        let mut best = None;
        let mut score = 0;
        for depth in 1..=self.limits.depth {
            // From depth 4 on, search a narrow window around the last score and widen it on a miss.
            let mut delta = ASPIRATION_WINDOW;
            let (mut low, mut high) = if depth >= 4 {
                (score - delta, score + delta)
            } else {
                (-INFINITY, INFINITY)
            };
            loop {
                score = self.negamax(pos, depth, low, high, 0, true);
                if self.stopped {
                    break;
                }
                if score <= low {
                    low = (score - delta).max(-INFINITY);
                } else if score >= high {
                    high = (score + delta).min(INFINITY);
                } else {
                    break;
                }
                delta *= 2;
            }
            // A stopped iteration is unfinished; the previous one's move stands.
            if self.stopped && best.is_some() {
                break;
            }
            // Stopped inside the first iteration before any root move was searched to the end.
            if self.pv_len[0] == 0 {
                self.pv[0][0] = pos.legal_moves()[0];
                self.pv_len[0] = 1;
                score = 0;
            }
            best = Some((self.pv[0][0], score));

            let elapsed = self.start.elapsed();
            let ms = elapsed.as_millis() as u64;
            let pv: Vec<String> = self.pv[0][..self.pv_len[0]]
                .iter()
                .map(Move::to_string)
                .collect();
            report(&format!(
                "info depth {depth} score {} nodes {} time {ms} nps {} pv {}",
                score_text(score),
                self.nodes,
                self.nodes * 1000 / ms.max(1),
                pv.join(" ")
            ));
            if self.stopped || self.limits.soft.is_some_and(|soft| elapsed >= soft) {
                break;
            }
        }
        best
    }

    fn negamax(
        &mut self,
        pos: &Position,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        allow_null: bool,
    ) -> i32 {
        self.pv_len[ply] = 0;
        if ply > 0 && (pos.halfmove >= 100 || self.repeated(pos)) {
            return 0;
        }
        let in_check = pos.in_check();
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.quiescence(pos, alpha, beta, ply);
        }
        self.nodes += 1;
        if self.nodes.is_multiple_of(1024) {
            self.check_limits();
        }
        if self.stopped {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.static_eval(pos, ply);
        }

        let pv_node = beta - alpha > 1;
        let excluded = self.excluded[ply];
        let singular_search = excluded != Move::default();
        let entry = self.tt.probe(pos.hash);
        // Without a table move the ordering is poor, so search a ply shallower; the next visit has the move.
        if depth >= IIR_DEPTH && entry.is_none_or(|e| e.best == Move::default()) {
            depth -= 1;
        }
        if !pv_node
            && !singular_search
            && let Some(e) = entry
            && e.depth as i32 >= depth
        {
            let score = from_tt(e.score as i32, ply);
            if e.bound == EXACT
                || (e.bound == LOWER && score >= beta)
                || (e.bound == UPPER && score <= alpha)
            {
                return score;
            }
        }

        let static_eval = if in_check {
            -INFINITY
        } else {
            self.static_eval(pos, ply)
        };
        if !pv_node && !in_check && !singular_search && beta.abs() < MATE_BOUND {
            if depth <= RFP_DEPTH && static_eval - RFP_MARGIN * depth >= beta {
                return static_eval;
            }
            // Passing is never legal, so a side with only pawns can be in zugzwang; skip the null move there.
            let pieces = pos.colors[pos.side] & !(pos.pieces[PAWN] | pos.pieces[KING]);
            if allow_null && depth >= 3 && static_eval >= beta && pieces != 0 {
                self.path.push(pos.hash);
                let null = &pos.make_null();
                if self.net.is_some() {
                    self.accumulators[ply + 1] = self.accumulators[ply];
                }
                let score = -self.negamax(
                    null,
                    depth - 1 - NULL_REDUCTION,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    false,
                );
                self.path.pop();
                if self.stopped {
                    return 0;
                }
                if score >= beta {
                    return if score >= MATE_BOUND { beta } else { score };
                }
            }
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        let mut keys = order_keys(pos, &moves, entry.map(|e| e.best));
        for (key, &m) in keys.iter_mut().zip(moves.iter()) {
            if *key == 0 {
                *key = if m == self.killers[ply][0] {
                    KILLER_KEY
                } else if m == self.killers[ply][1] {
                    KILLER_KEY - 1
                } else {
                    self.history[pos.side][m.from()][m.to()].min(KILLER_KEY - 2)
                };
            }
        }
        let alpha_start = alpha;
        let mut best = -INFINITY;
        let mut best_move = Move::default();
        let mut quiets = [Move::default(); 64];
        let mut quiet_count = 0;
        self.path.push(pos.hash);
        for i in 0..moves.len() {
            let m = pick(&mut moves, &mut keys, i);
            if m == excluded {
                continue;
            }
            let child = pos.make_move(m);
            let quiet = !m.is_capture() && m.promotion().is_none();
            // The table's move is searched a ply deeper when nothing else comes near its score: every other move
            // is searched at half depth against a bound just below it.
            let mut extension = 0;
            if !singular_search
                && ply > 0
                && depth >= SINGULAR_DEPTH
                && let Some(e) = entry
                && e.best == m
                && e.depth as i32 >= depth - 3
                && e.bound != UPPER
                && (e.score as i32).abs() < MATE_BOUND
            {
                let singular_beta = from_tt(e.score as i32, ply) - SINGULAR_MARGIN * depth;
                self.excluded[ply] = m;
                self.path.pop();
                let score = self.negamax(
                    pos,
                    (depth - 1) / 2,
                    singular_beta - 1,
                    singular_beta,
                    ply,
                    false,
                );
                self.path.push(pos.hash);
                self.excluded[ply] = Move::default();
                if self.stopped {
                    break;
                }
                extension = i32::from(score < singular_beta);
            }
            let new_depth = depth - 1 + extension;
            // Quiet moves far below alpha, or late in a long list at low depth, are skipped once one move has
            // shown the node is not lost.
            if !pv_node && !in_check && quiet && best > -MATE_BOUND && !child.in_check() {
                if depth <= LMP_DEPTH && quiet_count >= 3 + (depth * depth) as usize {
                    continue;
                }
                if depth <= FUTILITY_DEPTH
                    && static_eval + FUTILITY_BASE + FUTILITY_MARGIN * depth <= alpha
                {
                    continue;
                }
            }
            self.push_accumulator(ply, pos, &child, m);
            let score = if i == 0 {
                -self.negamax(&child, new_depth, -beta, -alpha, ply + 1, true)
            } else {
                // Late quiet moves get a shallower null-window search first; anything that beats alpha is
                // searched again at full depth, then with the full window.
                let reduction =
                    if depth >= 3 && i >= LMR_FULL_MOVES && quiet && !in_check && !child.in_check()
                    {
                        let killer = self.killers[ply].contains(&m);
                        (self.reductions[depth.min(63) as usize][i.min(63)]
                            - i32::from(pv_node)
                            - i32::from(killer))
                        .clamp(0, depth - 2)
                    } else {
                        0
                    };
                let mut score = -self.negamax(
                    &child,
                    new_depth - reduction,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    true,
                );
                if score > alpha && reduction > 0 {
                    score = -self.negamax(&child, new_depth, -alpha - 1, -alpha, ply + 1, true);
                }
                if score > alpha && score < beta {
                    score = -self.negamax(&child, new_depth, -beta, -alpha, ply + 1, true);
                }
                score
            };
            if self.stopped {
                break;
            }
            if score > best {
                best = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    self.update_pv(ply, m);
                    if score >= beta {
                        if quiet {
                            self.record_cutoff(ply, pos.side, m, depth, &quiets[..quiet_count]);
                        }
                        break;
                    }
                }
            }
            if quiet && quiet_count < quiets.len() {
                quiets[quiet_count] = m;
                quiet_count += 1;
            }
        }
        self.path.pop();
        if !self.stopped && !singular_search {
            let bound = if best >= beta {
                LOWER
            } else if best > alpha_start {
                EXACT
            } else {
                UPPER
            };
            self.tt
                .store(pos.hash, best_move, to_tt(best, ply), depth, bound);
        }
        best
    }

    // Captures and queen promotions until the position is quiet, standing pat on the static score unless
    // in check, where every evasion is searched.
    fn quiescence(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.nodes.is_multiple_of(1024) {
            self.check_limits();
        }
        if self.stopped {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.static_eval(pos, ply);
        }

        let in_check = pos.in_check();
        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        let mut keys = order_keys(pos, &moves, None);
        let mut best = if in_check {
            -INFINITY
        } else {
            self.static_eval(pos, ply)
        };
        if best >= beta {
            return best;
        }
        alpha = alpha.max(best);

        for i in 0..moves.len() {
            let m = pick(&mut moves, &mut keys, i);
            // Keys put every capture and queen promotion before the quiet moves.
            if !in_check && keys[i] == 0 {
                break;
            }
            if !in_check && m.promotion().is_none() && see(pos, m) < 0 {
                continue;
            }
            let child = pos.make_move(m);
            self.push_accumulator(ply, pos, &child, m);
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if score >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    // The cutoff move's history rises and that of the quiet moves tried before it falls, each change shrinking as
    // the score nears HISTORY_MAX so no entry runs away.
    fn record_cutoff(&mut self, ply: usize, side: usize, m: Move, depth: i32, failed: &[Move]) {
        if self.killers[ply][0] != m {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = m;
        }
        let bonus = (depth * depth).min(HISTORY_MAX / 4);
        let moves = std::iter::once((m, bonus)).chain(failed.iter().map(|&f| (f, -bonus)));
        for (mv, change) in moves {
            let score = &mut self.history[side][mv.from()][mv.to()];
            *score += change - *score * change.abs() / HISTORY_MAX;
        }
    }

    fn push_accumulator(&mut self, ply: usize, pos: &Position, child: &Position, m: Move) {
        if let Some(net) = self.net.as_deref() {
            let (done, next) = self.accumulators.split_at_mut(ply + 1);
            next[0].update(&done[ply], net, pos, child, m);
        }
    }

    fn static_eval(&self, pos: &Position, ply: usize) -> i32 {
        match self.net.as_deref() {
            Some(net) => net.evaluate(&self.accumulators[ply], pos),
            None => evaluate(pos),
        }
    }

    fn update_pv(&mut self, ply: usize, m: Move) {
        let child_len = self.pv_len[ply + 1];
        let (head, tail) = self.pv.split_at_mut(ply + 1);
        head[ply][0] = m;
        head[ply][1..=child_len].copy_from_slice(&tail[0][..child_len]);
        self.pv_len[ply] = child_len + 1;
    }

    // Twofold repetition since the last capture or pawn move. Positions with the same side to move sit
    // two, four, six plies back.
    fn repeated(&self, pos: &Position) -> bool {
        let n = self.path.len();
        let reach = (pos.halfmove as usize).min(n);
        (2..=reach)
            .step_by(2)
            .any(|back| self.path[n - back] == pos.hash)
    }

    fn check_limits(&mut self) {
        self.stopped = self.stop.load(Ordering::Relaxed)
            || self.nodes >= self.limits.nodes
            || self
                .limits
                .hard
                .is_some_and(|hard| self.start.elapsed() >= hard);
    }
}

// Captures by most valuable victim, then least valuable attacker; queen promotions next; quiet moves get 0.
fn order_key(pos: &Position, m: Move) -> i32 {
    if m.is_capture() {
        let victim = if m.flag() == Move::EN_PASSANT {
            PAWN
        } else {
            pos.board[m.to()] as usize % 6
        };
        let attacker = pos.board[m.from()] as usize % 6;
        CAPTURE_KEY + 10 * victim as i32 - attacker as i32
    } else if m.promotion() == Some(QUEEN) {
        PROMOTION_KEY
    } else {
        0
    }
}

fn order_keys(pos: &Position, moves: &[Move], tt_move: Option<Move>) -> [i32; 256] {
    let mut keys = [0; 256];
    for (key, &m) in keys.iter_mut().zip(moves) {
        *key = if Some(m) == tt_move {
            TT_KEY
        } else {
            order_key(pos, m)
        };
    }
    keys
}

// What the side to move gains by `m` and the recaptures on its square, each side free to stop: the swap
// algorithm of the Chess Programming Wiki. Pins and promotions are ignored.
fn see(pos: &Position, m: Move) -> i32 {
    let to = m.to();
    let mut occupied = pos.colors[WHITE] | pos.colors[BLACK];
    let mut gain = [0; 32];
    gain[0] = if m.flag() == Move::EN_PASSANT {
        occupied ^= 1 << (to ^ 8);
        SEE_VALUES[PAWN]
    } else if pos.board[to] == EMPTY {
        0
    } else {
        SEE_VALUES[pos.board[to] as usize % 6]
    };
    let diagonal = pos.pieces[BISHOP] | pos.pieces[QUEEN];
    let straight = pos.pieces[ROOK] | pos.pieces[QUEEN];
    let mut attackers = pos.attackers_to(to, occupied);
    let mut from = m.from();
    let mut kind = pos.board[from] as usize % 6;
    let mut side = pos.side;
    let mut d = 0;
    loop {
        // Gain for the side capturing at depth d + 1, kept only if a capturer is found.
        d += 1;
        gain[d] = SEE_VALUES[kind] - gain[d - 1];
        occupied ^= 1 << from;
        attackers |=
            bishop_attacks(to, occupied) & diagonal | rook_attacks(to, occupied) & straight;
        attackers &= occupied;
        side ^= 1;
        let ours = attackers & pos.colors[side];
        let Some(next) = (PAWN..=KING).find(|&k| ours & pos.pieces[k] != 0) else {
            break;
        };
        kind = next;
        from = squares(ours & pos.pieces[kind])
            .next()
            .expect("found above");
    }
    while d > 1 {
        d -= 1;
        gain[d - 1] = -(-gain[d - 1]).max(gain[d]);
    }
    gain[0]
}

// The table holds mate scores as distance from the stored node, the search as distance from the root.
fn to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_BOUND {
        score + ply as i32
    } else if score < -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

fn from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_BOUND {
        score - ply as i32
    } else if score < -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

// Swaps the best remaining move into place i; ties keep the earlier move.
fn pick(moves: &mut [Move], keys: &mut [i32], i: usize) -> Move {
    let mut best = i;
    for j in i + 1..moves.len() {
        if keys[j] > keys[best] {
            best = j;
        }
    }
    moves.swap(i, best);
    keys.swap(i, best);
    moves[i]
}

fn score_text(score: i32) -> String {
    if score > MATE_BOUND {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score < -MATE_BOUND {
        format!("mate -{}", (MATE + score) / 2)
    } else {
        format!("cp {score}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best_move(fen: &str, depth: i32) -> (Option<Move>, Vec<String>) {
        let pos = Position::from_fen(fen).unwrap();
        let mut lines = Vec::new();
        let limits = Limits {
            depth,
            ..Limits::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let best = Searcher::default().search(&pos, &[], limits, stop, &mut |line| {
            lines.push(line.to_string())
        });
        (best.map(|(m, _)| m), lines)
    }

    #[test]
    fn finds_mate_in_one() {
        let (best, lines) = best_move("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3);
        assert_eq!(best.map(|m| m.to_string()), Some("a1a8".to_string()));
        assert!(lines.last().unwrap().contains("score mate 1 "), "{lines:?}");
    }

    #[test]
    fn sees_mate_against_itself() {
        // Black's only move is Kg8, and Ra8 mates.
        let (_, lines) = best_move("7k/8/6K1/8/8/8/8/R7 b - - 0 1", 4);
        assert!(
            lines.last().unwrap().contains("score mate -1 "),
            "{lines:?}"
        );
    }

    #[test]
    fn no_move_when_mated_or_stalemated() {
        assert_eq!(
            best_move(
                "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
                3
            )
            .0,
            None
        );
        assert_eq!(best_move("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3).0, None);
    }

    #[test]
    fn wins_a_hanging_queen() {
        let (best, _) = best_move("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3);
        assert_eq!(best.map(|m| m.to_string()), Some("d1d5".to_string()));
    }

    #[test]
    fn mate_scores_move_between_root_and_node_distance() {
        for ply in [0, 1, 7, 40] {
            for score in [MATE - 3, -MATE + 5, 0, 250, -999] {
                assert_eq!(from_tt(to_tt(score, ply), ply), score);
            }
        }
        // Mated 5 plies below the root at a node 3 plies down is mated 2 plies below that node.
        assert_eq!(to_tt(-MATE + 5, 3), -MATE + 2);
        assert_eq!(from_tt(-MATE + 2, 6), -MATE + 8);
    }

    #[test]
    fn finds_mate_in_two_with_pruning_on() {
        // 1. Kb6 Kb8 2. Rh8 mate; there is no mate in one.
        let (_, lines) = best_move("k7/8/2K5/8/8/8/8/7R w - - 0 1", 6);
        assert!(lines.last().unwrap().contains("score mate 2 "), "{lines:?}");
    }

    #[test]
    fn quiescence_sees_the_recapture() {
        // Qxe5 wins a pawn at depth 1 by static count, but d6xe5 takes the queen back.
        let (best, lines) = best_move("4k3/8/3p4/4p3/8/8/8/4Q1K1 w - - 0 1", 1);
        assert_ne!(
            best.map(|m| m.to_string()),
            Some("e1e5".to_string()),
            "{lines:?}"
        );
    }

    #[test]
    fn repetition_counts_from_two_plies_back() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 6 10").unwrap();
        let mut searcher = Searcher {
            path: vec![1, 2, pos.hash],
            ..Searcher::default()
        };
        assert!(
            !searcher.repeated(&pos),
            "one ply back has the other side to move"
        );
        searcher.path = vec![pos.hash, 1];
        assert!(searcher.repeated(&pos));
        let fresh = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 10").unwrap();
        searcher.path = vec![fresh.hash, 1];
        assert!(
            !searcher.repeated(&fresh),
            "nothing repeats across a capture or pawn move"
        );
    }

    #[test]
    fn killers_keep_the_two_latest_cutoff_moves() {
        let mut searcher = Searcher::default();
        let (a, b) = (
            Move::new(12, 28, Move::DOUBLE_PUSH),
            Move::new(6, 21, Move::QUIET),
        );
        searcher.record_cutoff(3, 0, a, 4, &[]);
        searcher.record_cutoff(3, 0, a, 4, &[]);
        assert_eq!(
            searcher.killers[3],
            [a, Move::default()],
            "a repeat does not push the old killer out"
        );
        searcher.record_cutoff(3, 0, b, 2, &[a]);
        assert_eq!(searcher.killers[3], [b, a]);
        // 16 then 16 + 16 - 16 * 16 / 16384 = 32, then the failed move loses 4 - 32 * 4 / 16384 = 4.
        assert_eq!(searcher.history[0][12][28], 28);
        assert_eq!(searcher.history[0][6][21], 4);
        assert_eq!(
            searcher.killers[2],
            [Move::default(); 2],
            "other plies are untouched"
        );
    }

    #[test]
    fn a_second_search_reuses_the_table() {
        let pos = Position::from_fen(crate::board::STARTPOS).unwrap();
        let mut searcher = Searcher::default();
        let mut nodes = Vec::new();
        for _ in 0..2 {
            let limits = Limits {
                depth: 5,
                ..Limits::default()
            };
            searcher.search(
                &pos,
                &[],
                limits,
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            );
            nodes.push(searcher.nodes());
        }
        assert!(nodes[1] < nodes[0], "{nodes:?}");
    }

    #[test]
    fn searches_with_a_network_loaded() {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let mut searcher = Searcher::default();
        searcher.set_network(Some(Arc::new(crate::nnue::tests::random_network())));
        let limits = Limits {
            depth: 4,
            ..Limits::default()
        };
        let (best, _) = searcher
            .search(
                &pos,
                &[],
                limits,
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap();
        assert!(pos.legal_moves().contains(&best));
    }

    #[test]
    fn limits_hold_inside_the_first_iteration() {
        // Depth 1 here takes about 2,100 nodes, so a 1,000-node limit must stop the search inside it.
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let mut searcher = Searcher::default();
        let limits = Limits {
            nodes: 1_000,
            ..Limits::default()
        };
        let (best, _) = searcher
            .search(
                &pos,
                &[],
                limits,
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap();
        assert!(pos.legal_moves().contains(&best));
        assert!(searcher.nodes() < 2_000, "{}", searcher.nodes());
        let mut lines = Vec::new();
        Searcher::default().search(
            &pos,
            &[],
            Limits {
                nodes: 1_000,
                ..Limits::default()
            },
            Arc::new(AtomicBool::new(false)),
            &mut |l| lines.push(l.to_string()),
        );
        assert!(
            lines
                .iter()
                .all(|l| l.contains(" score cp ") || l.contains(" score mate ")),
            "{lines:?}"
        );
        assert!(!lines.iter().any(|l| l.contains("--")), "{lines:?}");
    }

    #[test]
    fn exchanges_are_counted_to_the_end() {
        let cases = [
            ("4k3/8/8/3r4/8/8/8/3RK3 w - - 0 1", "d1d5", 500),
            // The knight takes a pawn and the c6 pawn takes the knight.
            ("4k3/8/2p5/3p4/8/2N5/8/4K3 w - - 0 1", "c3d5", -200),
            // Queen takes pawn, bishop takes queen, the rook behind her takes the bishop.
            ("4k3/8/5b2/4p3/8/8/4Q3/4R1K1 w - - 0 1", "e2e5", -500),
            // Rook takes rook, bishop takes rook, the queen behind takes the bishop.
            ("4k3/8/5b2/4r3/8/8/4R3/4Q1K1 w - - 0 1", "e2e5", 300),
            // The king may take back a rook that nothing else defends, but not one the second rook guards.
            ("8/8/4k3/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", -400),
            ("8/8/4k3/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 100),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 100),
        ];
        for (fen, text, expected) in cases {
            let pos = Position::from_fen(fen).unwrap();
            let m = *pos
                .legal_moves()
                .iter()
                .find(|m| m.to_string() == text)
                .unwrap_or_else(|| panic!("{text} is not legal in {fen}"));
            assert_eq!(see(&pos, m), expected, "{fen} {text}");
        }
    }

    #[test]
    fn helpers_share_the_table_and_stop_with_the_main_search() {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let mut searcher = Searcher::new(4, 3, None);
        assert_eq!(searcher.helpers.len(), 2);
        assert!(
            searcher
                .helpers
                .iter()
                .all(|h| Arc::ptr_eq(&h.tt, &searcher.tt))
        );
        let limits = Limits {
            depth: 6,
            ..Limits::default()
        };
        let start = Instant::now();
        let (best, _) = searcher
            .search(
                &pos,
                &[],
                limits,
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap();
        assert!(pos.legal_moves().contains(&best));
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "{:?}",
            start.elapsed()
        );
        assert!(
            searcher.helpers.iter().all(|h| h.nodes() > 0),
            "every helper searched"
        );
    }

    #[test]
    fn an_excluded_move_is_left_out_of_the_search() {
        let pos = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let take = Move::new(3, 35, Move::CAPTURE);
        assert!(pos.legal_moves().contains(&take));
        let mut searcher = Searcher::default();
        let with = searcher.negamax(&pos, 3, -INFINITY, INFINITY, 1, false);
        assert_eq!(searcher.pv[1][0], take);
        searcher.excluded[1] = take;
        let without = searcher.negamax(&pos, 3, -INFINITY, INFINITY, 1, false);
        assert_ne!(searcher.pv[1][0], take);
        assert!(without < with - 500, "with {with}, without {without}");
    }

    #[test]
    fn node_limit_is_respected() {
        let pos = Position::from_fen(crate::board::STARTPOS).unwrap();
        let mut searcher = Searcher::default();
        let limits = Limits {
            nodes: 20_000,
            ..Limits::default()
        };
        let best = searcher.search(
            &pos,
            &[],
            limits,
            Arc::new(AtomicBool::new(false)),
            &mut |_| {},
        );
        assert!(best.is_some());
        assert!(searcher.nodes() < 20_000 + 1024, "{}", searcher.nodes());
    }
}
