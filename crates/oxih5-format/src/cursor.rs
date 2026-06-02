//! Bounds-checked sequential byte reader for HDF5 format parsing.

use oxih5_core::OxiH5Error;

/// Sequential, bounds-checked reader over a byte slice.
///
/// All `read_*` methods advance the internal position. Each returns
/// `Err` on out-of-bounds access instead of panicking.
pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Create a new cursor at position 0.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Create a cursor at an arbitrary starting position.
    pub fn at(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    /// Current byte position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Number of bytes remaining from the current position.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Seek to an absolute position. Seeking past the end is not an error;
    /// the next read will return `Err`.
    pub fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Advance the position by `n` bytes, returning `Err` if that would exceed bounds.
    pub fn skip(&mut self, n: usize) -> Result<(), OxiH5Error> {
        let new_pos = self.pos.checked_add(n).ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "cursor skip overflow: pos={} + n={}",
                self.pos, n
            ))
        })?;
        if new_pos > self.data.len() {
            return Err(OxiH5Error::Corrupted(format!(
                "cursor skip out of bounds: pos={} + n={} > len={}",
                self.pos, n, self.data.len()
            )));
        }
        self.pos = new_pos;
        Ok(())
    }

    /// Return a slice of `n` bytes at the current position and advance.
    pub fn read_slice(&mut self, n: usize) -> Result<&'a [u8], OxiH5Error> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "cursor slice overflow: pos={} + n={}",
                self.pos, n
            ))
        })?;
        let slice = self.data.get(self.pos..end).ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "cursor read_slice({n}) out of bounds: pos={}, len={}",
                self.pos,
                self.data.len()
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    /// Read 1 byte.
    pub fn read_u8(&mut self) -> Result<u8, OxiH5Error> {
        let b = self.data.get(self.pos).copied().ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "cursor read_u8 out of bounds: pos={}, len={}",
                self.pos,
                self.data.len()
            ))
        })?;
        self.pos += 1;
        Ok(b)
    }

    /// Read 2 bytes little-endian.
    pub fn read_u16_le(&mut self) -> Result<u16, OxiH5Error> {
        let s = self.read_slice(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    /// Read 3 bytes little-endian (24-bit field, returned as u32).
    pub fn read_u24_le(&mut self) -> Result<u32, OxiH5Error> {
        let s = self.read_slice(3)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], 0]))
    }

    /// Read 4 bytes little-endian.
    pub fn read_u32_le(&mut self) -> Result<u32, OxiH5Error> {
        let s = self.read_slice(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Read 8 bytes little-endian.
    pub fn read_u64_le(&mut self) -> Result<u64, OxiH5Error> {
        let s = self.read_slice(8)?;
        Ok(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }

    /// Read `width` bytes (1, 2, 4, or 8) as a little-endian u64.
    /// This is the generic backing method used by `read_offset_cur` / `read_length_cur`.
    pub fn read_uint_le(&mut self, width: usize) -> Result<u64, OxiH5Error> {
        let s = self.read_slice(width)?;
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(s);
        Ok(u64::from_le_bytes(buf))
    }

    /// Return the full underlying data slice.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u8_basic() {
        let data = [0x01, 0x02, 0x03];
        let mut c = Cursor::new(&data);
        assert_eq!(c.read_u8().unwrap(), 0x01);
        assert_eq!(c.pos(), 1);
        assert_eq!(c.read_u8().unwrap(), 0x02);
    }

    #[test]
    fn read_u8_oob_returns_err_not_panic() {
        let data = [0xAB];
        let mut c = Cursor::new(&data);
        assert!(c.read_u8().is_ok());
        assert!(c.read_u8().is_err()); // OOB -> Err, not panic
    }

    #[test]
    fn read_u16_le() {
        let data = [0x34, 0x12];
        let mut c = Cursor::new(&data);
        assert_eq!(c.read_u16_le().unwrap(), 0x1234);
    }

    #[test]
    fn read_u32_le() {
        let data = [0x78, 0x56, 0x34, 0x12];
        let mut c = Cursor::new(&data);
        assert_eq!(c.read_u32_le().unwrap(), 0x12345678);
    }

    #[test]
    fn read_u64_le() {
        let data = [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01];
        let mut c = Cursor::new(&data);
        assert_eq!(c.read_u64_le().unwrap(), 0x0102030405060708);
    }

    #[test]
    fn read_uint_le_widths() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut c = Cursor::new(&data);
        assert_eq!(c.read_uint_le(1).unwrap(), 0x01);
        c.seek(0);
        assert_eq!(c.read_uint_le(2).unwrap(), 0x0201);
        c.seek(0);
        assert_eq!(c.read_uint_le(4).unwrap(), 0x04030201);
        c.seek(0);
        assert_eq!(c.read_uint_le(8).unwrap(), 0x0807060504030201);
    }

    #[test]
    fn skip_advances_pos() {
        let data = [0u8; 16];
        let mut c = Cursor::new(&data);
        c.skip(8).unwrap();
        assert_eq!(c.pos(), 8);
    }

    #[test]
    fn skip_oob_returns_err() {
        let data = [0u8; 4];
        let mut c = Cursor::new(&data);
        assert!(c.skip(10).is_err());
    }

    #[test]
    fn seek_and_read() {
        let data = [0x00, 0xAB, 0xCD];
        let mut c = Cursor::new(&data);
        c.seek(1);
        assert_eq!(c.read_u8().unwrap(), 0xAB);
        assert_eq!(c.read_u8().unwrap(), 0xCD);
    }

    #[test]
    fn remaining_decrements() {
        let data = [0u8; 8];
        let mut c = Cursor::new(&data);
        assert_eq!(c.remaining(), 8);
        c.read_u32_le().unwrap();
        assert_eq!(c.remaining(), 4);
    }

    #[test]
    fn read_slice_returns_correct_bytes() {
        let data = [1, 2, 3, 4, 5];
        let mut c = Cursor::new(&data);
        let s = c.read_slice(3).unwrap();
        assert_eq!(s, &[1, 2, 3]);
        assert_eq!(c.pos(), 3);
    }

    #[test]
    fn empty_slice_read_oob() {
        let data = [];
        let mut c = Cursor::new(&data);
        assert!(c.read_u8().is_err());
    }
}
