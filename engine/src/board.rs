use crate::attacks::{KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks};
use crate::movegen::Move;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

pub const EMPTY: u8 = 12;

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_H: u64 = FILE_A << 7;
pub const RANK_1: u64 = 0xFF;
pub const RANK_3: u64 = RANK_1 << 16;
pub const RANK_6: u64 = RANK_1 << 40;
pub const RANK_8: u64 = RANK_1 << 56;

pub const WHITE_KINGSIDE: u8 = 1;
pub const WHITE_QUEENSIDE: u8 = 2;
pub const BLACK_KINGSIDE: u8 = 4;
pub const BLACK_QUEENSIDE: u8 = 8;

const PIECE_CHARS: &[u8; 12] = b"PNBRQKpnbrqk";

// A move from or to a1, e1, h1, a8, e8 or h8 loses the rights tied to that square.
const CASTLE_MASK: [u8; 64] = {
    let mut mask = [15; 64];
    mask[0] = 15 ^ WHITE_QUEENSIDE;
    mask[4] = 15 ^ WHITE_KINGSIDE ^ WHITE_QUEENSIDE;
    mask[7] = 15 ^ WHITE_KINGSIDE;
    mask[56] = 15 ^ BLACK_QUEENSIDE;
    mask[60] = 15 ^ BLACK_KINGSIDE ^ BLACK_QUEENSIDE;
    mask[63] = 15 ^ BLACK_KINGSIDE;
    mask
};

struct Keys {
    pieces: [[u64; 64]; 12],
    castling: [u64; 16],
    ep_file: [u64; 8],
    side: u64,
}

static KEYS: Keys = {
    let mut state = 0;
    let mut keys = Keys {
        pieces: [[0; 64]; 12],
        castling: [0; 16],
        ep_file: [0; 8],
        side: 0,
    };
    let mut i = 0;
    while i < 12 * 64 {
        keys.pieces[i / 64][i % 64] = splitmix64(&mut state);
        i += 1;
    }
    i = 0;
    while i < 16 {
        keys.castling[i] = splitmix64(&mut state);
        i += 1;
    }
    i = 0;
    while i < 8 {
        keys.ep_file[i] = splitmix64(&mut state);
        i += 1;
    }
    keys.side = splitmix64(&mut state);
    keys
};

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn piece(color: usize, kind: usize) -> u8 {
    (color * 6 + kind) as u8
}

pub fn squares(mut bb: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if bb == 0 {
            return None;
        }
        let sq = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        Some(sq)
    })
}

pub fn square_name(sq: usize) -> String {
    format!("{}{}", (b'a' + (sq % 8) as u8) as char, sq / 8 + 1)
}

pub fn parse_square(text: &str) -> Option<usize> {
    match text.as_bytes() {
        [file @ b'a'..=b'h', rank @ b'1'..=b'8'] => {
            Some(((rank - b'1') * 8 + (file - b'a')) as usize)
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position {
    pub(crate) pieces: [u64; 6],
    pub(crate) colors: [u64; 2],
    pub(crate) board: [u8; 64],
    pub(crate) side: usize,
    pub(crate) castling: u8,
    // Set only when a pawn of the side to move attacks the square, so equal positions hash equally.
    pub(crate) ep: Option<u8>,
    pub(crate) halfmove: u16,
    pub(crate) fullmove: u16,
    pub(crate) hash: u64,
}

impl Position {
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if !(4..=6).contains(&fields.len()) {
            return Err(format!(
                "FEN needs 4 to 6 fields, got {}: {fen:?}",
                fields.len()
            ));
        }
        let mut pos = Position {
            pieces: [0; 6],
            colors: [0; 2],
            board: [EMPTY; 64],
            side: WHITE,
            castling: 0,
            ep: None,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
        };

        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(format!(
                "FEN placement needs 8 ranks, got {}: {:?}",
                ranks.len(),
                fields[0]
            ));
        }
        for (i, rank_text) in ranks.iter().enumerate() {
            let rank = 7 - i;
            let mut file = 0;
            for c in rank_text.chars() {
                let (found, width) = match c.to_digit(10) {
                    Some(n @ 1..=8) => (None, n as usize),
                    Some(_) => return Err(format!("Bad digit {c:?} in FEN rank {rank_text:?}")),
                    None => match PIECE_CHARS.iter().position(|&p| p as char == c) {
                        Some(index) => (Some(index as u8), 1),
                        None => {
                            return Err(format!("Unknown piece {c:?} in FEN rank {rank_text:?}"));
                        }
                    },
                };
                if file + width > 8 {
                    return Err(format!("FEN rank {rank_text:?} is longer than 8 squares"));
                }
                if let Some(p) = found {
                    pos.put(p, rank * 8 + file);
                }
                file += width;
            }
            if file != 8 {
                return Err(format!("FEN rank {rank_text:?} does not cover 8 squares"));
            }
        }

        pos.side = match fields[1] {
            "w" => WHITE,
            "b" => BLACK,
            other => return Err(format!("FEN side to move must be w or b, got {other:?}")),
        };

        if fields[2] != "-" {
            for c in fields[2].chars() {
                let (right, king_sq, rook_sq, color) = match c {
                    'K' => (WHITE_KINGSIDE, 4, 7, WHITE),
                    'Q' => (WHITE_QUEENSIDE, 4, 0, WHITE),
                    'k' => (BLACK_KINGSIDE, 60, 63, BLACK),
                    'q' => (BLACK_QUEENSIDE, 60, 56, BLACK),
                    _ => return Err(format!("Unknown castling right {c:?} in {:?}", fields[2])),
                };
                if pos.castling & right != 0 {
                    return Err(format!("Castling right {c:?} repeated in {:?}", fields[2]));
                }
                if pos.board[king_sq] != piece(color, KING)
                    || pos.board[rook_sq] != piece(color, ROOK)
                {
                    return Err(format!(
                        "Castling right {c:?} needs its king and rook on their home squares"
                    ));
                }
                pos.castling |= right;
            }
        }

        if fields[3] != "-" {
            let ep = parse_square(fields[3])
                .ok_or_else(|| format!("Bad en passant square {:?}", fields[3]))?;
            let mover = pos.side ^ 1;
            let (ep_rank, pawn_sq, origin) = if mover == BLACK {
                (5, ep - 8, ep + 8)
            } else {
                (2, ep + 8, ep - 8)
            };
            if ep / 8 != ep_rank {
                return Err(format!(
                    "En passant square {:?} is not on the rank a double push passes",
                    fields[3]
                ));
            }
            if pos.board[pawn_sq] != piece(mover, PAWN)
                || pos.board[ep] != EMPTY
                || pos.board[origin] != EMPTY
            {
                return Err(format!(
                    "En passant square {:?} has no pawn that just made a double push",
                    fields[3]
                ));
            }
            if PAWN_ATTACKS[mover][ep] & pos.pieces[PAWN] & pos.colors[pos.side] != 0 {
                pos.ep = Some(ep as u8);
            }
        }

        if let Some(text) = fields.get(4) {
            pos.halfmove = text
                .parse()
                .map_err(|_| format!("Bad halfmove clock {text:?}"))?;
        }
        if let Some(text) = fields.get(5) {
            pos.fullmove = match text.parse() {
                Ok(n) if n > 0 => n,
                _ => return Err(format!("Bad fullmove number {text:?}")),
            };
        }

        for (color, name) in [(WHITE, "white"), (BLACK, "black")] {
            if (pos.pieces[KING] & pos.colors[color]).count_ones() != 1 {
                return Err(format!("FEN needs exactly one {name} king"));
            }
        }
        if pos.pieces[PAWN] & (RANK_1 | RANK_8) != 0 {
            return Err("FEN has a pawn on the first or last rank".to_string());
        }
        let their_king = (pos.pieces[KING] & pos.colors[pos.side ^ 1]).trailing_zeros() as usize;
        if pos.attackers_to(their_king, pos.colors[WHITE] | pos.colors[BLACK])
            & pos.colors[pos.side]
            != 0
        {
            return Err("FEN leaves the side not to move in check".to_string());
        }

        pos.hash = pos.compute_hash();
        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let p = self.board[rank * 8 + file];
                if p == EMPTY {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    fen.push((b'0' + empty) as char);
                    empty = 0;
                }
                fen.push(PIECE_CHARS[p as usize] as char);
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push_str(if self.side == WHITE { " w " } else { " b " });
        if self.castling == 0 {
            fen.push('-');
        }
        for (right, c) in [
            (WHITE_KINGSIDE, 'K'),
            (WHITE_QUEENSIDE, 'Q'),
            (BLACK_KINGSIDE, 'k'),
            (BLACK_QUEENSIDE, 'q'),
        ] {
            if self.castling & right != 0 {
                fen.push(c);
            }
        }
        let ep = self
            .ep
            .map_or("-".to_string(), |sq| square_name(sq as usize));
        format!("{fen} {ep} {} {}", self.halfmove, self.fullmove)
    }

    pub(crate) fn attackers_to(&self, sq: usize, occupied: u64) -> u64 {
        (PAWN_ATTACKS[WHITE][sq] & self.pieces[PAWN] & self.colors[BLACK])
            | (PAWN_ATTACKS[BLACK][sq] & self.pieces[PAWN] & self.colors[WHITE])
            | (KNIGHT_ATTACKS[sq] & self.pieces[KNIGHT])
            | (KING_ATTACKS[sq] & self.pieces[KING])
            | (bishop_attacks(sq, occupied) & (self.pieces[BISHOP] | self.pieces[QUEEN]))
            | (rook_attacks(sq, occupied) & (self.pieces[ROOK] | self.pieces[QUEEN]))
    }

    pub fn side(&self) -> usize {
        self.side
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn in_check(&self) -> bool {
        let king = (self.pieces[KING] & self.colors[self.side]).trailing_zeros() as usize;
        let occupied = self.colors[WHITE] | self.colors[BLACK];
        self.attackers_to(king, occupied) & self.colors[self.side ^ 1] != 0
    }

    // The side to move passes. Used only inside the search, where the halfmove clock restarts so no repetition is
    // counted across the pass.
    pub fn make_null(&self) -> Position {
        let mut next = *self;
        next.hash ^= KEYS.side;
        if let Some(ep) = self.ep {
            next.hash ^= KEYS.ep_file[ep as usize % 8];
            next.ep = None;
        }
        next.side ^= 1;
        next.halfmove = 0;
        next
    }

    pub fn make_move(&self, m: Move) -> Position {
        let mut next = *self;
        let us = self.side;
        let (from, to, flag) = (m.from(), m.to(), m.flag());

        next.hash ^= KEYS.side;
        if let Some(ep) = self.ep {
            next.hash ^= KEYS.ep_file[ep as usize % 8];
            next.ep = None;
        }
        next.halfmove += 1;

        if m.is_capture() {
            next.take(if flag == Move::EN_PASSANT { to ^ 8 } else { to });
            next.halfmove = 0;
        }
        let moved = next.take(from);
        next.put(m.promotion().map_or(moved, |kind| piece(us, kind)), to);

        if moved as usize % 6 == PAWN {
            next.halfmove = 0;
            let ep = (from + to) / 2;
            if flag == Move::DOUBLE_PUSH
                && PAWN_ATTACKS[us][ep] & next.pieces[PAWN] & next.colors[us ^ 1] != 0
            {
                next.ep = Some(ep as u8);
                next.hash ^= KEYS.ep_file[ep % 8];
            }
        }
        if flag == Move::KING_CASTLE {
            let rook = next.take(to + 1);
            next.put(rook, to - 1);
        } else if flag == Move::QUEEN_CASTLE {
            let rook = next.take(to - 2);
            next.put(rook, to + 1);
        }

        let rights = self.castling & CASTLE_MASK[from] & CASTLE_MASK[to];
        next.hash ^= KEYS.castling[self.castling as usize] ^ KEYS.castling[rights as usize];
        next.castling = rights;
        if us == BLACK {
            next.fullmove += 1;
        }
        next.side = us ^ 1;
        next
    }

    fn put(&mut self, p: u8, sq: usize) {
        let bit = 1 << sq;
        self.pieces[p as usize % 6] |= bit;
        self.colors[p as usize / 6] |= bit;
        self.board[sq] = p;
        self.hash ^= KEYS.pieces[p as usize][sq];
    }

    fn take(&mut self, sq: usize) -> u8 {
        let p = self.board[sq];
        let bit = 1 << sq;
        self.pieces[p as usize % 6] ^= bit;
        self.colors[p as usize / 6] ^= bit;
        self.board[sq] = EMPTY;
        self.hash ^= KEYS.pieces[p as usize][sq];
        p
    }

    fn compute_hash(&self) -> u64 {
        let mut hash = KEYS.castling[self.castling as usize];
        for (sq, &p) in self.board.iter().enumerate() {
            if p != EMPTY {
                hash ^= KEYS.pieces[p as usize][sq];
            }
        }
        if let Some(ep) = self.ep {
            hash ^= KEYS.ep_file[ep as usize % 8];
        }
        if self.side == BLACK {
            hash ^= KEYS.side;
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FENS: [&str; 6] = [
        STARTPOS,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    ];

    #[test]
    fn fen_round_trips() {
        for fen in FENS {
            assert_eq!(Position::from_fen(fen).unwrap().to_fen(), fen);
        }
    }

    #[test]
    fn en_passant_without_a_capturer_is_dropped() {
        let pos =
            Position::from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
                .unwrap();
        assert_eq!(pos.ep, None);
        assert_eq!(
            pos.to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        );
    }

    #[test]
    fn bad_fens_are_rejected() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
        let cases = [
            ("one two three".to_string(), "4 to 6 fields"),
            (format!("{STARTPOS} extra"), "4 to 6 fields"),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1".to_string(),
                "8 ranks",
            ),
            (
                "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                "Bad digit",
            ),
            (
                "rnbqkbnr/ppppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                "longer than 8",
            ),
            (
                "rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                "does not cover",
            ),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1".to_string(),
                "Unknown piece",
            ),
            (format!("{start} x KQkq - 0 1"), "side to move"),
            (format!("{start} w KQkx - 0 1"), "Unknown castling"),
            (format!("{start} w KKkq - 0 1"), "repeated"),
            (
                "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                "home squares",
            ),
            (format!("{start} w KQkq e9 0 1"), "Bad en passant"),
            (format!("{start} w KQkq e5 0 1"), "not on the rank"),
            (format!("{start} w KQkq e6 0 1"), "no pawn"),
            (format!("{start} w KQkq - x 1"), "halfmove"),
            (format!("{start} w KQkq - 0 0"), "fullmove"),
            (
                "rnbqqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1".to_string(),
                "one black king",
            ),
            (
                "4k3/8/8/8/8/8/8/P3K3 w - - 0 1".to_string(),
                "pawn on the first",
            ),
            (
                "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1".to_string(),
                "not to move in check",
            ),
        ];
        for (fen, why) in cases {
            let err = Position::from_fen(&fen).unwrap_err();
            assert!(err.contains(why), "{fen:?} gave {err:?}, expected {why:?}");
        }
    }

    #[test]
    fn a_null_move_flips_the_side_and_keeps_the_hash_exact() {
        let pos = Position::from_fen(FENS[5]).unwrap();
        let null = pos.make_null();
        assert_eq!((null.side, null.ep), (pos.side ^ 1, None));
        assert_eq!(null.hash, null.compute_hash());
        assert_eq!(
            null.make_null().hash,
            Position { ep: None, ..pos }.compute_hash()
        );
    }

    #[test]
    fn zobrist_keys_are_distinct() {
        let mut all: Vec<u64> = KEYS
            .pieces
            .iter()
            .flatten()
            .chain(&KEYS.castling)
            .chain(&KEYS.ep_file)
            .copied()
            .collect();
        all.push(KEYS.side);
        let count = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), count);
        assert!(!all.contains(&0));
    }

    // Every position reached must keep its incremental hash exact and survive a FEN round trip.
    fn walk(pos: &Position, depth: u32) {
        assert_eq!(pos.hash, pos.compute_hash(), "{}", pos.to_fen());
        assert_eq!(
            Position::from_fen(&pos.to_fen()).as_ref(),
            Ok(pos),
            "{}",
            pos.to_fen()
        );
        if depth > 0 {
            for &m in pos.legal_moves().iter() {
                walk(&pos.make_move(m), depth - 1);
            }
        }
    }

    #[test]
    fn positions_reached_stay_consistent() {
        for fen in FENS {
            walk(&Position::from_fen(fen).unwrap(), 3);
        }
    }
}
