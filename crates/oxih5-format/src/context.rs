use oxih5_core::OxiH5Error;

/// Carries file-level metadata through parse functions.
#[derive(Debug, Clone, Copy)]
pub struct ParseContext {
    pub size_of_offsets: u8,
    pub size_of_lengths: u8,
    pub base_address: u64,
}

impl ParseContext {
    /// Create a new ParseContext.
    pub fn new(size_of_offsets: u8, size_of_lengths: u8, base_address: u64) -> Self {
        Self {
            size_of_offsets,
            size_of_lengths,
            base_address,
        }
    }

    /// Default context for soo=8, sol=8 (covers all current files).
    pub fn default_v0() -> Self {
        Self {
            size_of_offsets: 8,
            size_of_lengths: 8,
            base_address: 0,
        }
    }

    /// Read an offset (soo bytes) from data at pos.
    pub fn read_offset(&self, data: &[u8], pos: usize) -> Result<u64, OxiH5Error> {
        self.read_int(data, pos, self.size_of_offsets as usize)
    }

    /// Read a length (sol bytes) from data at pos.
    pub fn read_length(&self, data: &[u8], pos: usize) -> Result<u64, OxiH5Error> {
        self.read_int(data, pos, self.size_of_lengths as usize)
    }

    /// Read an integer of arbitrary byte width (1/2/4/8) from data at pos.
    ///
    /// Useful when the width is determined at runtime (e.g. link-name-length field).
    pub fn read_int_generic(
        &self,
        data: &[u8],
        pos: usize,
        size: usize,
    ) -> Result<u64, OxiH5Error> {
        self.read_int(data, pos, size)
    }

    fn read_int(&self, data: &[u8], pos: usize, size: usize) -> Result<u64, OxiH5Error> {
        let bytes = data.get(pos..pos + size).ok_or_else(|| {
            OxiH5Error::Format(format!(
                "read_int: pos {} size {} out of bounds (data len {})",
                pos,
                size,
                data.len()
            ))
        })?;
        Ok(match size {
            1 => bytes[0] as u64,
            2 => u16::from_le_bytes([bytes[0], bytes[1]]) as u64,
            4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
            8 => u64::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("read_int u64".into()))?,
            ),
            _ => {
                return Err(OxiH5Error::Format(format!(
                    "read_int: unsupported size {}",
                    size
                )))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_v0() {
        let ctx = ParseContext::default_v0();
        assert_eq!(ctx.size_of_offsets, 8);
        assert_eq!(ctx.size_of_lengths, 8);
        assert_eq!(ctx.base_address, 0);
    }

    #[test]
    fn test_read_offset_u64() {
        let ctx = ParseContext::default_v0();
        let mut data = vec![0u8; 16];
        let val: u64 = 0xDEAD_BEEF_1234_5678;
        data[0..8].copy_from_slice(&val.to_le_bytes());
        assert_eq!(ctx.read_offset(&data, 0).unwrap(), val);
    }

    #[test]
    fn test_read_length_out_of_bounds() {
        let ctx = ParseContext::default_v0();
        let data = vec![0u8; 4];
        assert!(ctx.read_length(&data, 0).is_err());
    }

    #[test]
    fn test_read_int_various_sizes() {
        let ctx4 = ParseContext::new(4, 4, 0);
        let data = vec![0x78, 0x56, 0x34, 0x12];
        assert_eq!(ctx4.read_offset(&data, 0).unwrap(), 0x1234_5678);

        let ctx2 = ParseContext::new(2, 2, 0);
        let data2 = vec![0xAB, 0xCD];
        assert_eq!(ctx2.read_offset(&data2, 0).unwrap(), 0xCDAB);

        let ctx1 = ParseContext::new(1, 1, 0);
        let data1 = vec![0xFF];
        assert_eq!(ctx1.read_offset(&data1, 0).unwrap(), 255);
    }
}
