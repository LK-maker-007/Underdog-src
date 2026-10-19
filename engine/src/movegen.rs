use std::fmt;
use std::ops::Deref;

use crate::attacks::{
    BETWEEN, KING_ATTACKS, KNIGHT_ATTACKS, LINE, PAWN_ATTACKS, bishop_attacks, rook_attacks,
};
use crate::board::{
    BISHOP, BLACK_KINGSIDE, BLACK_QUEENSIDE, FILE_A, FILE_H, KING, KNIGHT, PAWN, Position, QUEEN,
    RANK_1, RANK_3, RANK_6, RANK_8, ROOK, WHITE, WHITE_KINGSIDE, WHITE_QUEENSIDE, square_name,
    squares,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Move(u16);

impl Move {
    pub const QUIET: u16 = 0;
    pub const DOUBLE_PUSH: u16 = 1;
    pub const KING_CASTLE: u16 = 2;
    pub const QUEEN_CASTLE: u16 = 3;
    pub const CAPTURE: u16 = 4;
    pub const EN_PASSANT: u16 = 5;
    // The low two bits of a promotion flag pick the piece: knight, bishop, rook, queen.
    pub const PROMOTION: u16 = 8;

    pub const fn new(from: usize, to: usize, flag: u16) -> Move {
        Move(from as u16 | ((to as u16) << 6) | (flag << 12))
    }

    pub const fn from(self) -> usize {
        (self.0 & 63) as usize
    }

    pub const fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }

    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    pub const fn is_capture(self) -> bool {
        self.flag() & Move::CAPTURE != 0
    }

    pub const fn promotion(self) -> Option<usize> {
        if self.flag() & Move::PROMOTION != 0 {
            Some(KNIGHT + (self.flag() & 3) as usize)
        } else {
            None
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", square_name(self.from()), square_name(self.to()))?;
        if let Some(kind) = self.promotion() {
            write!(f, "{}", ['n', 'b', 'r', 'q'][kind - KNIGHT])?;
        }
        Ok(())
    }
}

// No reachable position has more than 218 legal moves.
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl Default for MoveList {
    fn default() -> MoveList {
        MoveList {
            moves: [Move::default(); 256],
            len: 0,
        }
    }
}

impl Deref for MoveList {
    type Target = [Move];

    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl MoveList {
    fn push(&mut self, from: usize, to: usize, flag: u16) {
        self.moves[self.len] = Move::new(from, to, flag);
        self.len += 1;
    }

    fn push_targets(&mut self, from: usize, targets: u64, theirs: u64) {
        for to in squares(targets) {
            self.push(
                from,
                to,
                if theirs & (1 << to) != 0 {
                    Move::CAPTURE
                } else {
                    Move::QUIET
                },
            );
        }
    }
}

fn shift(bb: u64, by: i32) -> u64 {
    if by > 0 { bb << by } else { bb >> -by }
}

impl Position {
    pub fn legal_moves(&self) -> MoveList {
        let mut list = MoveList::default();
        let us = self.side;
        let ours = self.colors[us];
        let theirs = self.colors[us ^ 1];
        let occupied = ours | theirs;
        let king = (self.pieces[KING] & ours).trailing_zeros() as usize;
        let checkers = self.attackers_to(king, occupied) & theirs;
        let danger = self.attacked_by(us ^ 1, occupied ^ (1 << king));

        list.push_targets(king, KING_ATTACKS[king] & !ours & !danger, theirs);
        if checkers.count_ones() > 1 {
            return list;
        }

        let check_mask = if checkers == 0 {
            !0
        } else {
            BETWEEN[king][checkers.trailing_zeros() as usize] | checkers
        };
        let rook_like = (self.pieces[ROOK] | self.pieces[QUEEN]) & theirs;
        let bishop_like = (self.pieces[BISHOP] | self.pieces[QUEEN]) & theirs;
        // Rays from the king through our own pieces find every slider that could pin one of them.
        let mut pinned = 0;
        for sniper in squares(
            (rook_attacks(king, theirs) & rook_like) | (bishop_attacks(king, theirs) & bishop_like),
        ) {
            let blockers = BETWEEN[king][sniper] & occupied;
            if blockers.count_ones() == 1 {
                pinned |= blockers & ours;
            }
        }

        let targets = !ours & check_mask;
        for from in squares(self.pieces[KNIGHT] & ours & !pinned) {
            list.push_targets(from, KNIGHT_ATTACKS[from] & targets, theirs);
        }
        for from in squares((self.pieces[BISHOP] | self.pieces[QUEEN]) & ours) {
            let pin_ray = if pinned & (1 << from) != 0 {
                LINE[king][from]
            } else {
                !0
            };
            list.push_targets(
                from,
                bishop_attacks(from, occupied) & targets & pin_ray,
                theirs,
            );
        }
        for from in squares((self.pieces[ROOK] | self.pieces[QUEEN]) & ours) {
            let pin_ray = if pinned & (1 << from) != 0 {
                LINE[king][from]
            } else {
                !0
            };
            list.push_targets(
                from,
                rook_attacks(from, occupied) & targets & pin_ray,
                theirs,
            );
        }
        self.pawn_moves(&mut list, king, pinned, check_mask);

        if checkers == 0 {
            let (kingside, queenside) = if us == WHITE {
                (WHITE_KINGSIDE, WHITE_QUEENSIDE)
            } else {
                (BLACK_KINGSIDE, BLACK_QUEENSIDE)
            };
            // Holding a right means the king and that rook are still on their home squares.
            if self.castling & kingside != 0 {
                let path = (1 << (king + 1)) | (1 << (king + 2));
                if (occupied | danger) & path == 0 {
                    list.push(king, king + 2, Move::KING_CASTLE);
                }
            }
            if self.castling & queenside != 0 {
                let path = (1 << (king - 1)) | (1 << (king - 2));
                if occupied & (path | (1 << (king - 3))) == 0 && danger & path == 0 {
                    list.push(king, king - 2, Move::QUEEN_CASTLE);
                }
            }
        }
        list
    }

    fn pawn_moves(&self, list: &mut MoveList, king: usize, pinned: u64, check_mask: u64) {
        let us = self.side;
        let theirs = self.colors[us ^ 1];
        let occupied = self.colors[0] | self.colors[1];
        let pawns = self.pieces[PAWN] & self.colors[us];
        let (up, double_rank) = if us == WHITE {
            (8, RANK_3)
        } else {
            (-8, RANK_6)
        };

        let single = shift(pawns, up) & !occupied;
        let double = shift(single & double_rank, up) & !occupied;
        let left = shift(pawns & !FILE_A, up - 1) & theirs;
        let right = shift(pawns & !FILE_H, up + 1) & theirs;
        for (targets, delta, flag) in [
            (single, up, Move::QUIET),
            (double, 2 * up, Move::DOUBLE_PUSH),
            (left, up - 1, Move::CAPTURE),
            (right, up + 1, Move::CAPTURE),
        ] {
            for to in squares(targets & check_mask) {
                let from = (to as i32 - delta) as usize;
                if pinned & (1 << from) != 0 && LINE[king][from] & (1 << to) == 0 {
                    continue;
                }
                if (RANK_1 | RANK_8) & (1 << to) != 0 {
                    for kind in 0..4 {
                        list.push(from, to, flag | Move::PROMOTION | kind);
                    }
                } else {
                    list.push(from, to, flag);
                }
            }
        }

        if let Some(ep) = self.ep {
            let ep = ep as usize;
            let captured = ep ^ 8;
            if check_mask & ((1 << ep) | (1 << captured)) == 0 {
                return;
            }
            let rook_like = (self.pieces[ROOK] | self.pieces[QUEEN]) & theirs;
            let bishop_like = (self.pieces[BISHOP] | self.pieces[QUEEN]) & theirs;
            for from in squares(PAWN_ATTACKS[us ^ 1][ep] & pawns) {
                // Both pawns leave their squares at once, which can expose the king along the rank.
                let after = (occupied ^ (1 << from) ^ (1 << captured)) | (1 << ep);
                if rook_attacks(king, after) & rook_like == 0
                    && bishop_attacks(king, after) & bishop_like == 0
                {
                    list.push(from, ep, Move::EN_PASSANT);
                }
            }
        }
    }

    fn attacked_by(&self, color: usize, occupied: u64) -> u64 {
        let pieces = self.colors[color];
        let pawns = self.pieces[PAWN] & pieces;
        let mut attacked = if color == WHITE {
            ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9)
        } else {
            ((pawns & !FILE_A) >> 9) | ((pawns & !FILE_H) >> 7)
        };
        for sq in squares(self.pieces[KNIGHT] & pieces) {
            attacked |= KNIGHT_ATTACKS[sq];
        }
        for sq in squares((self.pieces[BISHOP] | self.pieces[QUEEN]) & pieces) {
            attacked |= bishop_attacks(sq, occupied);
        }
        for sq in squares((self.pieces[ROOK] | self.pieces[QUEEN]) & pieces) {
            attacked |= rook_attacks(sq, occupied);
        }
        attacked | KING_ATTACKS[(self.pieces[KING] & pieces).trailing_zeros() as usize]
    }
}

pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = pos.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .iter()
        .map(|&m| perft(&pos.make_move(m), depth - 1))
        .sum()
}
