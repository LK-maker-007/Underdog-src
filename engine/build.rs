use std::fmt::Write;
use std::path::Path;
use std::{env, fs};

#[path = "src/rays.rs"]
mod rays;

use rays::{BISHOP_DIRS, ROOK_DIRS, slide};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/rays.rs");

    let mut out = String::new();
    let mut table = Vec::new();
    // A fixed seed keeps the generated tables identical from build to build.
    let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
    for (name, dirs) in [("ROOK_MAGICS", &ROOK_DIRS), ("BISHOP_MAGICS", &BISHOP_DIRS)] {
        writeln!(out, "static {name}: [Magic; 64] = [").unwrap();
        for sq in 0..64 {
            let mask = relevant_mask(sq, dirs);
            let (magic, attacks) = find_magic(sq, mask, dirs, &mut rng);
            writeln!(
                out,
                "    Magic {{ mask: {mask:#x}, magic: {magic:#x}, offset: {}, shift: {} }},",
                table.len(),
                64 - mask.count_ones()
            )
            .unwrap();
            table.extend(attacks);
        }
        out.push_str("];\n");
    }
    writeln!(out, "static SLIDER_ATTACKS: [u64; {}] = [", table.len()).unwrap();
    for attacks in &table {
        writeln!(out, "{attacks:#x},").unwrap();
    }
    out.push_str("];\n");

    let mut between = [[0u64; 64]; 64];
    let mut line = [[0u64; 64]; 64];
    for a in 0..64 {
        for b in 0..64 {
            for dirs in [&ROOK_DIRS, &BISHOP_DIRS] {
                if a != b && slide(a, 0, dirs) & (1 << b) != 0 {
                    between[a][b] = slide(a, 1 << b, dirs) & slide(b, 1 << a, dirs);
                    line[a][b] = (slide(a, 0, dirs) & slide(b, 0, dirs)) | (1 << a) | (1 << b);
                }
            }
        }
    }
    for (name, rows) in [("BETWEEN", &between), ("LINE", &line)] {
        writeln!(out, "pub static {name}: [[u64; 64]; 64] = [").unwrap();
        for row in rows {
            out.push('[');
            for bb in row {
                write!(out, "{bb:#x},").unwrap();
            }
            out.push_str("],\n");
        }
        out.push_str("];\n");
    }

    let path = Path::new(&env::var("OUT_DIR").unwrap()).join("tables.rs");
    fs::write(path, out).unwrap();
}

// Squares whose occupancy changes the slider's attacks: the rays without their last square.
fn relevant_mask(sq: usize, dirs: &[(i32, i32)]) -> u64 {
    let mut mask = 0;
    for &(df, dr) in dirs {
        let (mut file, mut rank) = ((sq % 8) as i32 + df, (sq / 8) as i32 + dr);
        while (0..8).contains(&(file + df)) && (0..8).contains(&(rank + dr)) {
            mask |= 1u64 << (rank * 8 + file);
            file += df;
            rank += dr;
        }
    }
    mask
}

fn find_magic(sq: usize, mask: u64, dirs: &[(i32, i32)], rng: &mut u64) -> (u64, Vec<u64>) {
    let mut occupancies = Vec::new();
    let mut subset = 0u64;
    loop {
        occupancies.push((subset, slide(sq, subset, dirs)));
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }
    let shift = 64 - mask.count_ones();
    let mut table = vec![0u64; 1 << mask.count_ones()];
    let mut filled = vec![0u32; table.len()];
    let mut attempt = 0;
    loop {
        attempt += 1;
        let magic = next(rng) & next(rng) & next(rng);
        let mut ok = true;
        for &(occupied, attacks) in &occupancies {
            let index = (occupied.wrapping_mul(magic) >> shift) as usize;
            if filled[index] != attempt {
                filled[index] = attempt;
                table[index] = attacks;
            } else if table[index] != attacks {
                ok = false;
                break;
            }
        }
        if ok {
            return (magic, table);
        }
    }
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}
