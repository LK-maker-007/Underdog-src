use std::sync::atomic::{AtomicU64, Ordering};

use crate::movegen::Move;

pub const EXACT: u8 = 0;
pub const LOWER: u8 = 1;
pub const UPPER: u8 = 2;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Entry {
    pub best: Move,
    pub score: i16,
    pub depth: i8,
    pub bound: u8,
}

impl Entry {
    fn pack(self) -> u64 {
        let best = self.best.from() | self.best.to() << 6 | usize::from(self.best.flag()) << 12;
        best as u64
            | u64::from(self.score as u16) << 16
            | u64::from(self.depth as u8) << 32
            | u64::from(self.bound) << 40
    }

    fn unpack(data: u64) -> Entry {
        Entry {
            best: Move::new(
                (data & 63) as usize,
                (data >> 6 & 63) as usize,
                (data >> 12 & 15) as u16,
            ),
            score: (data >> 16) as u16 as i16,
            depth: (data >> 32) as u8 as i8,
            bound: (data >> 40) as u8,
        }
    }
}

// Shared by every search thread without locks. A slot holds the key XORed with the data, then the data, so a slot
// torn by two threads writing at once fails the key check and reads as empty.
pub struct Table {
    slots: Vec<[AtomicU64; 2]>,
}

impl Table {
    pub fn new(megabytes: usize) -> Table {
        let count = (megabytes << 20) / size_of::<[AtomicU64; 2]>();
        Table {
            slots: (0..count.max(1)).map(|_| Default::default()).collect(),
        }
    }

    // Maps the hash onto the table by its high bits, so the size need not be a power of two.
    fn index(&self, hash: u64) -> usize {
        ((hash as u128 * self.slots.len() as u128) >> 64) as usize
    }

    pub fn probe(&self, hash: u64) -> Option<Entry> {
        let [check, data] = &self.slots[self.index(hash)];
        let data = data.load(Ordering::Relaxed);
        (check.load(Ordering::Relaxed) ^ data == hash).then(|| Entry::unpack(data))
    }

    pub fn store(&self, hash: u64, best: Move, score: i32, depth: i32, bound: u8) {
        let data = Entry {
            best,
            score: score as i16,
            depth: depth as i8,
            bound,
        }
        .pack();
        let [check, slot] = &self.slots[self.index(hash)];
        check.store(hash ^ data, Ordering::Relaxed);
        slot.store(data, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_finds_by_full_key() {
        let table = Table::new(1);
        let m = Move::new(12, 28, Move::DOUBLE_PUSH);
        table.store(0xDEAD_BEEF_1234_5678, m, -250, 7, LOWER);
        let entry = table.probe(0xDEAD_BEEF_1234_5678).unwrap();
        assert_eq!(
            (entry.best, entry.score, entry.depth, entry.bound),
            (m, -250, 7, LOWER)
        );
        assert_eq!(
            table.index(0xDEAD_BEEF_1234_5678),
            table.index(0xDEAD_BEEF_1234_5679)
        );
        assert_eq!(
            table.probe(0xDEAD_BEEF_1234_5679),
            None,
            "same slot, different key"
        );
    }

    #[test]
    fn every_index_is_in_range() {
        let table = Table::new(1);
        for hash in [0, 1, u64::MAX, u64::MAX / 3, 0x8000_0000_0000_0000] {
            assert!(table.index(hash) < table.slots.len());
        }
    }
}
