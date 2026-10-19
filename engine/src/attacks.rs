struct Magic {
    mask: u64,
    magic: u64,
    offset: usize,
    shift: u32,
}

include!(concat!(env!("OUT_DIR"), "/tables.rs"));

pub static KNIGHT_ATTACKS: [u64; 64] = leaper_table(&[
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]);
pub static KING_ATTACKS: [u64; 64] = leaper_table(&[
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]);
// Indexed by the pawn's colour: the squares a pawn on each square attacks.
pub static PAWN_ATTACKS: [[u64; 64]; 2] = [
    leaper_table(&[(-1, 1), (1, 1)]),
    leaper_table(&[(-1, -1), (1, -1)]),
];

#[inline]
pub fn rook_attacks(sq: usize, occupied: u64) -> u64 {
    lookup(&ROOK_MAGICS[sq], occupied)
}

#[inline]
pub fn bishop_attacks(sq: usize, occupied: u64) -> u64 {
    lookup(&BISHOP_MAGICS[sq], occupied)
}

#[inline]
fn lookup(m: &Magic, occupied: u64) -> u64 {
    SLIDER_ATTACKS[m.offset + ((occupied & m.mask).wrapping_mul(m.magic) >> m.shift) as usize]
}

const fn leaper_table(steps: &[(i32, i32)]) -> [u64; 64] {
    let mut table = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let mut i = 0;
        while i < steps.len() {
            let file = (sq % 8) as i32 + steps[i].0;
            let rank = (sq / 8) as i32 + steps[i].1;
            if file >= 0 && file < 8 && rank >= 0 && rank < 8 {
                table[sq] |= 1 << (rank * 8 + file);
            }
            i += 1;
        }
        sq += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rays::{BISHOP_DIRS, ROOK_DIRS, slide};

    #[test]
    fn magic_lookup_matches_ray_walk() {
        let mut state = 0x1234_5678_9ABC_DEF1_u64;
        for sq in 0..64 {
            for _ in 0..2000 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                for occupied in [state, state & state.rotate_left(23) & state.rotate_left(41)] {
                    assert_eq!(
                        rook_attacks(sq, occupied),
                        slide(sq, occupied, &ROOK_DIRS),
                        "rook {sq} {occupied:#x}"
                    );
                    assert_eq!(
                        bishop_attacks(sq, occupied),
                        slide(sq, occupied, &BISHOP_DIRS),
                        "bishop {sq} {occupied:#x}"
                    );
                }
            }
        }
    }
}
