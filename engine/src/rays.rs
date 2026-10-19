pub const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

pub fn slide(sq: usize, occupied: u64, dirs: &[(i32, i32)]) -> u64 {
    let mut attacks = 0;
    for &(df, dr) in dirs {
        let (mut file, mut rank) = ((sq % 8) as i32 + df, (sq / 8) as i32 + dr);
        while (0..8).contains(&file) && (0..8).contains(&rank) {
            let bit = 1u64 << (rank * 8 + file);
            attacks |= bit;
            if occupied & bit != 0 {
                break;
            }
            file += df;
            rank += dr;
        }
    }
    attacks
}
