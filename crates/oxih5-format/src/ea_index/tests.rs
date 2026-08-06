//! Synthetic extensible arrays, encoded exactly the way libhdf5 does.
//!
//! [`EaSpec::build`] lays out a whole file — header, index block, super blocks,
//! data blocks and (when the creation parameters call for it) data-block pages
//! — from a list of per-element chunk addresses.  The geometry it uses is the
//! same `sblk_info` recurrence the parser derives independently, so a test that
//! round-trips through it exercises the encoding rather than the parser's own
//! arithmetic reflected back at itself.

use super::*;
use crate::ea_index::header::EA_HEADER_LEN;

/// Creation parameters of a synthetic extensible array.
#[derive(Debug, Clone, Copy)]
struct EaSpec {
    /// 0 = unfiltered chunk (bare address), 1 = filtered chunk.
    client: u8,
    /// Width of the stored-size field for `client == 1`.
    size_len: usize,
    max_nelmts_bits: u8,
    idx_blk_elmts: u64,
    data_blk_min_elmts: u64,
    sup_blk_min_data_ptrs: u64,
    max_dblk_page_nelmts_bits: u8,
}

impl Default for EaSpec {
    /// libhdf5's own defaults (`H5D_EARRAY_*` in `H5Dprivate.h`).
    fn default() -> Self {
        EaSpec {
            client: 0,
            size_len: 8,
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            data_blk_min_elmts: 16,
            sup_blk_min_data_ptrs: 4,
            max_dblk_page_nelmts_bits: 10,
        }
    }
}

/// One array element: an allocated chunk, or an unwritten slot.
#[derive(Debug, Clone, Copy)]
struct Elem {
    address: u64,
    size: u64,
    filter_mask: u32,
}

impl EaSpec {
    fn element_size(&self) -> usize {
        match self.client {
            0 => 8,
            _ => 8 + self.size_len + 4,
        }
    }

    fn arr_off(&self) -> usize {
        (self.max_nelmts_bits as usize).div_ceil(8)
    }

    fn nsblks_total(&self) -> usize {
        (self.max_nelmts_bits as u32 - self.data_blk_min_elmts.trailing_zeros() + 1) as usize
    }

    fn iblk_nsblks(&self) -> usize {
        2 * self.sup_blk_min_data_ptrs.trailing_zeros() as usize
    }

    fn ndblk_addrs(&self) -> usize {
        2 * (self.sup_blk_min_data_ptrs as usize - 1)
    }

    fn nsblk_addrs(&self) -> usize {
        self.nsblks_total() - self.iblk_nsblks()
    }

    fn page_nelmts(&self) -> u64 {
        1u64 << self.max_dblk_page_nelmts_bits
    }

    /// `(ndblks, dblk_nelmts, start_idx)` per super block.
    fn sblk_info(&self) -> Vec<(u64, u64, u64)> {
        let mut out = Vec::new();
        let mut start = 0u64;
        for u in 0..self.nsblks_total() {
            let ndblks = 1u64 << (u / 2);
            let dblk_nelmts = (1u64 << u.div_ceil(2)) * self.data_blk_min_elmts;
            out.push((ndblks, dblk_nelmts, start));
            start += ndblks * dblk_nelmts;
        }
        out
    }

    fn encode_element(&self, buf: &mut [u8], at: usize, elem: Option<Elem>) {
        let esz = self.element_size();
        match elem {
            None => buf[at..at + esz].fill(0xFF),
            Some(e) => {
                buf[at..at + 8].copy_from_slice(&e.address.to_le_bytes());
                if self.client != 0 {
                    let raw = e.size.to_le_bytes();
                    buf[at + 8..at + 8 + self.size_len].copy_from_slice(&raw[..self.size_len]);
                    let m = at + 8 + self.size_len;
                    buf[m..m + 4].copy_from_slice(&e.filter_mask.to_le_bytes());
                }
            }
        }
    }

    /// Build a whole file containing this array plus `elems`.
    ///
    /// Returns the file bytes and the extensible-array header address.
    fn build(&self, elems: &[Option<Elem>]) -> (Vec<u8>, u64) {
        let esz = self.element_size();
        let arr_off = self.arr_off();
        let info = self.sblk_info();
        let n = elems.len() as u64;

        // ---- Plan the layout ------------------------------------------------
        // Data blocks needed to cover element indices idx_blk_elmts..n.
        struct DblkPlan {
            u: usize,
            k: u64,
            first_index: u64,
            nelmts: u64,
            addr: usize,
            paged: bool,
        }
        let mut sblk_plan: Vec<(usize, usize)> = Vec::new(); // (super block index, addr)
        let mut dblk_plan: Vec<DblkPlan> = Vec::new();

        let ib_len = 14
            + (self.idx_blk_elmts as usize) * esz
            + (self.ndblk_addrs() + self.nsblk_addrs()) * 8
            + 4;
        let mut cursor = EA_HEADER_LEN + ib_len;

        let plan_dblk = |cursor: &mut usize, plan: &mut Vec<DblkPlan>, u: usize, k: u64| {
            let (_, dblk_nelmts, start_idx) = info[u];
            let first_index = self.idx_blk_elmts + start_idx + k * dblk_nelmts;
            if first_index >= n {
                return;
            }
            let paged = dblk_nelmts > self.page_nelmts();
            let size = if paged {
                let npages = dblk_nelmts.div_ceil(self.page_nelmts()) as usize;
                14 + arr_off + 4 + npages * (self.page_nelmts() as usize * esz + 4)
            } else {
                14 + arr_off + dblk_nelmts as usize * esz + 4
            };
            plan.push(DblkPlan {
                u,
                k,
                first_index,
                nelmts: dblk_nelmts,
                addr: *cursor,
                paged,
            });
            *cursor += size;
        };

        for (u, &(ndblks, _, _)) in info.iter().enumerate().take(self.iblk_nsblks()) {
            for k in 0..ndblks {
                plan_dblk(&mut cursor, &mut dblk_plan, u, k);
            }
        }
        for j in 0..self.nsblk_addrs() {
            let u = self.iblk_nsblks() + j;
            let (ndblks, dblk_nelmts, start_idx) = info[u];
            if self.idx_blk_elmts + start_idx >= n {
                break;
            }
            let paged = dblk_nelmts > self.page_nelmts();
            let page_init_bytes = if paged {
                ndblks as usize * (dblk_nelmts.div_ceil(self.page_nelmts()) as usize).div_ceil(8)
            } else {
                0
            };
            let sblk_addr = cursor;
            cursor += 14 + arr_off + page_init_bytes + ndblks as usize * 8 + 4;
            sblk_plan.push((u, sblk_addr));
            for k in 0..ndblks {
                plan_dblk(&mut cursor, &mut dblk_plan, u, k);
            }
        }

        let mut buf = vec![0u8; cursor];

        // ---- Header ---------------------------------------------------------
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 0;
        buf[5] = self.client;
        buf[6] = esz as u8;
        buf[7] = self.max_nelmts_bits;
        buf[8] = self.idx_blk_elmts as u8;
        buf[9] = self.data_blk_min_elmts as u8;
        buf[10] = self.sup_blk_min_data_ptrs as u8;
        buf[11] = self.max_dblk_page_nelmts_bits;
        buf[52..60].copy_from_slice(&n.to_le_bytes());
        buf[60..68].copy_from_slice(&(EA_HEADER_LEN as u64).to_le_bytes());

        // ---- Index block ----------------------------------------------------
        let ib = EA_HEADER_LEN;
        buf[ib..ib + 4].copy_from_slice(b"EAIB");
        buf[ib + 4] = 0;
        buf[ib + 5] = self.client;
        buf[ib + 6..ib + 14].copy_from_slice(&0u64.to_le_bytes());
        for i in 0..self.idx_blk_elmts {
            let at = ib + 14 + (i as usize) * esz;
            self.encode_element(&mut buf, at, elems.get(i as usize).copied().flatten());
        }
        let dblk_addr_area = ib + 14 + self.idx_blk_elmts as usize * esz;
        for i in 0..self.ndblk_addrs() {
            buf[dblk_addr_area + i * 8..dblk_addr_area + i * 8 + 8]
                .copy_from_slice(&u64::MAX.to_le_bytes());
        }
        let sblk_addr_area = dblk_addr_area + self.ndblk_addrs() * 8;
        for i in 0..self.nsblk_addrs() {
            buf[sblk_addr_area + i * 8..sblk_addr_area + i * 8 + 8]
                .copy_from_slice(&u64::MAX.to_le_bytes());
        }

        // Fill in the index block's data-block address slots in order.
        let mut slot = 0usize;
        for (u, &(ndblks, _, _)) in info.iter().enumerate().take(self.iblk_nsblks()) {
            for k in 0..ndblks {
                if let Some(p) = dblk_plan.iter().find(|p| p.u == u && p.k == k) {
                    let at = dblk_addr_area + slot * 8;
                    buf[at..at + 8].copy_from_slice(&(p.addr as u64).to_le_bytes());
                }
                slot += 1;
            }
        }
        for (u, sblk_addr) in &sblk_plan {
            let j = u - self.iblk_nsblks();
            let at = sblk_addr_area + j * 8;
            buf[at..at + 8].copy_from_slice(&(*sblk_addr as u64).to_le_bytes());
        }

        // ---- Super blocks ---------------------------------------------------
        for (u, sblk_addr) in &sblk_plan {
            let (ndblks, dblk_nelmts, start_idx) = info[*u];
            let sa = *sblk_addr;
            buf[sa..sa + 4].copy_from_slice(b"EASB");
            buf[sa + 4] = 0;
            buf[sa + 5] = self.client;
            buf[sa + 6..sa + 14].copy_from_slice(&0u64.to_le_bytes());
            let off_bytes = start_idx.to_le_bytes();
            buf[sa + 14..sa + 14 + arr_off].copy_from_slice(&off_bytes[..arr_off]);
            let paged = dblk_nelmts > self.page_nelmts();
            let npages = dblk_nelmts.div_ceil(self.page_nelmts());
            let per_block = (npages as usize).div_ceil(8);
            let page_init_bytes = if paged {
                ndblks as usize * per_block
            } else {
                0
            };
            let bitmap_at = sa + 14 + arr_off;
            let addrs_at = bitmap_at + page_init_bytes;
            for k in 0..ndblks {
                let at = addrs_at + (k as usize) * 8;
                match dblk_plan.iter().find(|p| p.u == *u && p.k == k) {
                    Some(p) => {
                        buf[at..at + 8].copy_from_slice(&(p.addr as u64).to_le_bytes());
                        if paged {
                            // Mark every page that holds at least one element.
                            for page in 0..npages {
                                if p.first_index + page * self.page_nelmts() >= n {
                                    break;
                                }
                                let bit = k * npages + page;
                                buf[bitmap_at + (bit / 8) as usize] |= 1 << (7 - (bit % 8));
                            }
                        }
                    }
                    None => buf[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes()),
                }
            }
        }

        // ---- Data blocks ----------------------------------------------------
        for p in &dblk_plan {
            let a = p.addr;
            buf[a..a + 4].copy_from_slice(b"EADB");
            buf[a + 4] = 0;
            buf[a + 5] = self.client;
            buf[a + 6..a + 14].copy_from_slice(&0u64.to_le_bytes());
            let off_bytes = (p.first_index - self.idx_blk_elmts).to_le_bytes();
            buf[a + 14..a + 14 + arr_off].copy_from_slice(&off_bytes[..arr_off]);
            if p.paged {
                let page_span = self.page_nelmts() as usize * esz + 4;
                let pages_start = a + 14 + arr_off + 4;
                for page in 0..p.nelmts.div_ceil(self.page_nelmts()) {
                    let po = pages_start + (page as usize) * page_span;
                    for i in 0..self.page_nelmts() {
                        let idx = p.first_index + page * self.page_nelmts() + i;
                        let at = po + (i as usize) * esz;
                        self.encode_element(
                            &mut buf,
                            at,
                            elems.get(idx as usize).copied().flatten(),
                        );
                    }
                }
            } else {
                let base = a + 14 + arr_off;
                for i in 0..p.nelmts {
                    let idx = p.first_index + i;
                    let at = base + (i as usize) * esz;
                    self.encode_element(&mut buf, at, elems.get(idx as usize).copied().flatten());
                }
            }
        }

        (buf, 0)
    }
}

/// Chunk addresses `0x10000 + i * 0x100`, all allocated.
fn dense_elems(n: usize) -> Vec<Option<Elem>> {
    (0..n)
        .map(|i| {
            Some(Elem {
                address: 0x1_0000 + (i as u64) * 0x100,
                size: 0,
                filter_mask: 0,
            })
        })
        .collect()
}

fn geom_1d<'a>(chunk: &'a [u64], dataset: &'a [u64]) -> EaGeometry<'a> {
    EaGeometry {
        chunk_dims: chunk,
        dataset_dims: dataset,
        max_dims: None,
        chunk_bytes: 64,
    }
}

// ---------------------------------------------------------------------------
// Header / structural validation
// ---------------------------------------------------------------------------

#[test]
fn ea_rejects_bad_signature() {
    let buf = vec![0u8; 128];
    let err = parse_extensible_array(&buf, 0, &geom_1d(&[4], &[16])).expect_err("must reject");
    assert!(
        matches!(err, OxiH5Error::Format(ref m) if m.contains("signature")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ea_without_index_block_is_empty() {
    let spec = EaSpec::default();
    let (mut buf, addr) = spec.build(&dense_elems(0));
    buf.resize(EA_HEADER_LEN.max(buf.len()), 0);
    buf[60..68].copy_from_slice(&u64::MAX.to_le_bytes());
    let records =
        parse_extensible_array(&buf, addr, &geom_1d(&[4], &[16])).expect("undefined index block");
    assert!(records.is_empty(), "{records:?}");
}

#[test]
fn ea_rejects_wrong_header_back_pointer() {
    let spec = EaSpec::default();
    let (mut buf, addr) = spec.build(&dense_elems(4));
    // Corrupt the index block's back-pointer to the header.
    let ib = EA_HEADER_LEN;
    buf[ib + 6..ib + 14].copy_from_slice(&0xDEADu64.to_le_bytes());
    let err = parse_extensible_array(&buf, addr, &geom_1d(&[4], &[16])).expect_err("must reject");
    assert!(
        matches!(err, OxiH5Error::Format(ref m) if m.contains("back-points")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ea_rejects_super_block_with_wrong_element_offset() {
    let spec = EaSpec::default();
    // 300 elements reach super block 4, whose addresses live in the index block
    // as a secondary-block pointer.
    let elems = dense_elems(300);
    let (mut buf, addr) = spec.build(&elems);
    let easb = buf
        .windows(4)
        .position(|w| w == b"EASB")
        .expect("a secondary block must have been emitted");
    let arr_off = spec.arr_off();
    buf[easb + 14..easb + 14 + arr_off].fill(0x7F);
    let err = parse_extensible_array(&buf, addr, &geom_1d(&[1], &[300])).expect_err("must reject");
    assert!(
        matches!(err, OxiH5Error::Format(ref m) if m.contains("element offset")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn ea_rejects_unknown_client_id() {
    let spec = EaSpec::default();
    let (mut buf, addr) = spec.build(&dense_elems(4));
    buf[5] = 2; // H5EA_CLS_TEST_ID — never written to a real file.
    let err = parse_extensible_array(&buf, addr, &geom_1d(&[4], &[16])).expect_err("must reject");
    assert!(
        matches!(err, OxiH5Error::NotImplemented(ref m) if m.contains("client ID")),
        "unexpected error: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Element traversal
// ---------------------------------------------------------------------------

/// Only the index block's inline elements: the shape of a dataset with fewer
/// chunks than `idx_blk_elmts`.
#[test]
fn ea_inline_elements_only() {
    let spec = EaSpec::default();
    let elems = dense_elems(4);
    let (buf, addr) = spec.build(&elems);
    let records =
        parse_extensible_array(&buf, addr, &geom_1d(&[3], &[10])).expect("inline elements");
    assert_eq!(records.len(), 4);
    for (i, rec) in records.iter().enumerate() {
        assert_eq!(rec.address, 0x1_0000 + (i as u64) * 0x100);
        assert_eq!(rec.offsets, vec![i as u64 * 3]);
        assert_eq!(rec.size, 64, "unfiltered records take the whole-chunk size");
        assert_eq!(rec.filter_mask, 0);
    }
}

/// The 125-chunk shape libhdf5 produces for `shape=(500,), chunks=(4,)`:
/// four inline elements plus five data blocks of 16/32/32/32/64 elements, all
/// addressed straight from the index block.
#[test]
fn ea_index_block_data_blocks() {
    let spec = EaSpec::default();
    let elems = dense_elems(125);
    let (buf, addr) = spec.build(&elems);
    let records =
        parse_extensible_array(&buf, addr, &geom_1d(&[4], &[500])).expect("data block elements");
    assert_eq!(records.len(), 125);
    let mut seen: Vec<u64> = records.iter().map(|r| r.offsets[0]).collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..125).map(|i| i * 4).collect::<Vec<_>>());
    for rec in &records {
        let i = rec.offsets[0] / 4;
        assert_eq!(rec.address, 0x1_0000 + i * 0x100, "element {i} misplaced");
    }
}

/// Beyond super block 3 the data-block addresses move into secondary blocks.
#[test]
fn ea_secondary_block_traversal() {
    let spec = EaSpec::default();
    let elems = dense_elems(400);
    let (buf, addr) = spec.build(&elems);
    assert!(
        buf.windows(4).any(|w| w == b"EASB"),
        "the fixture must exercise a secondary block"
    );
    let records =
        parse_extensible_array(&buf, addr, &geom_1d(&[1], &[400])).expect("secondary blocks");
    assert_eq!(records.len(), 400);
    for rec in &records {
        let i = rec.offsets[0];
        assert_eq!(rec.address, 0x1_0000 + i * 0x100, "element {i} misplaced");
    }
}

/// Unallocated slots are skipped without disturbing the elements after them.
#[test]
fn ea_skips_unallocated_slots() {
    let spec = EaSpec::default();
    let mut elems = dense_elems(40);
    elems[0] = None;
    elems[5] = None;
    elems[20] = None;
    let (buf, addr) = spec.build(&elems);
    let records = parse_extensible_array(&buf, addr, &geom_1d(&[1], &[40])).expect("sparse array");
    assert_eq!(records.len(), 37);
    let offsets: Vec<u64> = records.iter().map(|r| r.offsets[0]).collect();
    assert!(!offsets.contains(&0));
    assert!(!offsets.contains(&5));
    assert!(!offsets.contains(&20));
    assert!(offsets.contains(&4));
    assert!(offsets.contains(&21));
}

/// A filtered array stores address + variable-width stored size + filter mask.
#[test]
fn ea_filtered_elements_carry_size_and_mask() {
    let spec = EaSpec {
        client: 1,
        size_len: 8,
        ..EaSpec::default()
    };
    let elems: Vec<Option<Elem>> = (0..25u64)
        .map(|i| {
            Some(Elem {
                address: 0x1_0000 + i * 0x100,
                size: 29 + i,
                filter_mask: if i % 3 == 0 { 0 } else { 1 },
            })
        })
        .collect();
    let (buf, addr) = spec.build(&elems);
    let records =
        parse_extensible_array(&buf, addr, &geom_1d(&[8], &[200])).expect("filtered array");
    assert_eq!(records.len(), 25);
    for rec in &records {
        let i = rec.offsets[0] / 8;
        assert_eq!(rec.size, 29 + i as u32, "stored size for element {i}");
        assert_eq!(rec.filter_mask, if i % 3 == 0 { 0 } else { 1 });
    }
}

/// A narrower stored-size field (libhdf5 sizes it from the chunk size) must be
/// read from the header's element size, not assumed to be 8 bytes.
#[test]
fn ea_filtered_narrow_size_field() {
    let spec = EaSpec {
        client: 1,
        size_len: 2,
        ..EaSpec::default()
    };
    let elems: Vec<Option<Elem>> = (0..6u64)
        .map(|i| {
            Some(Elem {
                address: 0x2_0000 + i * 0x40,
                size: 300 + i,
                filter_mask: 0,
            })
        })
        .collect();
    let (buf, addr) = spec.build(&elems);
    let records = parse_extensible_array(&buf, addr, &geom_1d(&[2], &[12])).expect("narrow size");
    assert_eq!(records.len(), 6);
    for rec in &records {
        let i = rec.offsets[0] / 2;
        assert_eq!(rec.size, 300 + i as u32);
    }
}

/// A data block bigger than one page is split into pages, each with its own
/// checksum, and the owning super block holds the "page initialised" bits.
#[test]
fn ea_paged_data_blocks() {
    // These creation parameters page the first *super-block-owned* data block
    // (super block 3, 8 elements per block, 4 per page) while leaving the
    // index-block-owned ones unpaged.
    let spec = EaSpec {
        max_nelmts_bits: 8,
        idx_blk_elmts: 2,
        data_blk_min_elmts: 2,
        sup_blk_min_data_ptrs: 2,
        max_dblk_page_nelmts_bits: 2,
        ..EaSpec::default()
    };
    let elems = dense_elems(30);
    let (buf, addr) = spec.build(&elems);
    let records = parse_extensible_array(&buf, addr, &geom_1d(&[1], &[30])).expect("paged blocks");
    assert_eq!(records.len(), 30, "every element must survive paging");
    let mut offsets: Vec<u64> = records.iter().map(|r| r.offsets[0]).collect();
    offsets.sort_unstable();
    assert_eq!(offsets, (0..30).collect::<Vec<_>>());
    for rec in &records {
        let i = rec.offsets[0];
        assert_eq!(rec.address, 0x1_0000 + i * 0x100, "element {i} misplaced");
    }
}

/// An uninitialised page holds allocator leftovers, so its bit being clear must
/// keep the parser away from it entirely.
#[test]
fn ea_uninitialised_pages_are_not_decoded() {
    let spec = EaSpec {
        max_nelmts_bits: 8,
        idx_blk_elmts: 2,
        data_blk_min_elmts: 2,
        sup_blk_min_data_ptrs: 2,
        max_dblk_page_nelmts_bits: 2,
        ..EaSpec::default()
    };
    // 18 elements stop part-way into the paged data block (elements 16..23), so
    // its first page is written and its second is allocated but never
    // initialised.
    let elems = dense_elems(18);
    let (mut buf, addr) = spec.build(&elems);

    // Poison exactly the second page of the last data block: a parser that
    // ignored the page-init bitmask would decode 0x5555… as chunk addresses.
    let last_dblk = (0..buf.len() - 4)
        .rev()
        .find(|&i| &buf[i..i + 4] == b"EADB")
        .expect("a paged data block must have been emitted");
    let esz = spec.element_size();
    let page_span = spec.page_nelmts() as usize * esz + 4;
    let page1 = last_dblk + 14 + spec.arr_off() + 4 + page_span;
    buf[page1..page1 + page_span].fill(0x55);

    let records = parse_extensible_array(&buf, addr, &geom_1d(&[1], &[18])).expect("paged");
    assert_eq!(records.len(), 18, "{records:?}");
    let mut offsets: Vec<u64> = records.iter().map(|r| r.offsets[0]).collect();
    offsets.sort_unstable();
    assert_eq!(offsets, (0..18).collect::<Vec<_>>());
}

// ---------------------------------------------------------------------------
// Element index → chunk offset (the swizzled grid)
// ---------------------------------------------------------------------------

/// Rank 2 with the unlimited dimension first: plain row-major over the grid
/// derived from the maximum dimensions.
#[test]
fn ea_2d_unlimited_dim0_offsets() {
    let spec = EaSpec::default();
    let elems = dense_elems(6);
    let (buf, addr) = spec.build(&elems);
    let chunk = [2u64, 2];
    let dataset = [6u64, 4];
    let max = [u64::MAX, 4u64];
    let records = parse_extensible_array(
        &buf,
        addr,
        &EaGeometry {
            chunk_dims: &chunk,
            dataset_dims: &dataset,
            max_dims: Some(&max),
            chunk_bytes: 16,
        },
    )
    .expect("2-D EA");
    let offsets: Vec<Vec<u64>> = records.iter().map(|r| r.offsets.clone()).collect();
    assert_eq!(
        offsets,
        vec![
            vec![0, 0],
            vec![0, 2],
            vec![2, 0],
            vec![2, 2],
            vec![4, 0],
            vec![4, 2]
        ]
    );
}

/// A larger maximum in a *fixed* dimension widens the grid, so element indices
/// skip: chunk (2, 0) is element 4, not element 2.
#[test]
fn ea_2d_grid_comes_from_max_dims_not_current_dims() {
    let spec = EaSpec::default();
    let mut elems: Vec<Option<Elem>> = vec![None; 10];
    for (i, slot) in elems.iter_mut().enumerate() {
        if [0usize, 1, 4, 5, 8, 9].contains(&i) {
            *slot = Some(Elem {
                address: 0x1_0000 + (i as u64) * 0x100,
                size: 0,
                filter_mask: 0,
            });
        }
    }
    let (buf, addr) = spec.build(&elems);
    let chunk = [2u64, 2];
    let dataset = [6u64, 4];
    let max = [u64::MAX, 8u64];
    let records = parse_extensible_array(
        &buf,
        addr,
        &EaGeometry {
            chunk_dims: &chunk,
            dataset_dims: &dataset,
            max_dims: Some(&max),
            chunk_bytes: 16,
        },
    )
    .expect("2-D EA with a wide maximum");
    let offsets: Vec<Vec<u64>> = records.iter().map(|r| r.offsets.clone()).collect();
    assert_eq!(
        offsets,
        vec![
            vec![0, 0],
            vec![0, 2],
            vec![2, 0],
            vec![2, 2],
            vec![4, 0],
            vec![4, 2]
        ]
    );
}

/// Rank 3 with the unlimited dimension last.  `H5VM_swizzle_coords` *rotates*
/// it to the front (0, 1, 2 → 2, 0, 1); a swap of positions 0 and 2 would give
/// a different, wrong answer here.
#[test]
fn ea_3d_unlimited_last_dim_rotates_not_swaps() {
    let spec = EaSpec::default();
    let elems = dense_elems(12);
    let (buf, addr) = spec.build(&elems);
    let chunk = [2u64, 2, 2];
    let dataset = [4u64, 6, 4];
    let max = [4u64, 6, u64::MAX];
    let records = parse_extensible_array(
        &buf,
        addr,
        &EaGeometry {
            chunk_dims: &chunk,
            dataset_dims: &dataset,
            max_dims: Some(&max),
            chunk_bytes: 32,
        },
    )
    .expect("3-D EA");
    let offsets: Vec<Vec<u64>> = records.iter().map(|r| r.offsets.clone()).collect();
    // Element order is (unlim, dim0, dim1) slowest → fastest.
    assert_eq!(
        offsets,
        vec![
            vec![0, 0, 0],
            vec![0, 2, 0],
            vec![0, 4, 0],
            vec![2, 0, 0],
            vec![2, 2, 0],
            vec![2, 4, 0],
            vec![0, 0, 2],
            vec![0, 2, 2],
            vec![0, 4, 2],
            vec![2, 0, 2],
            vec![2, 2, 2],
            vec![2, 4, 2],
        ]
    );
}

/// Rank ≥ 2 without maximum dimensions cannot be located, and must say so
/// rather than guess that dimension 0 is the unlimited one.
#[test]
fn ea_2d_without_max_dims_is_rejected() {
    let spec = EaSpec::default();
    let (buf, addr) = spec.build(&dense_elems(4));
    let chunk = [2u64, 2];
    let dataset = [4u64, 4];
    let err = parse_extensible_array(
        &buf,
        addr,
        &EaGeometry {
            chunk_dims: &chunk,
            dataset_dims: &dataset,
            max_dims: None,
            chunk_bytes: 16,
        },
    )
    .expect_err("must reject");
    assert!(
        matches!(err, OxiH5Error::Format(ref m) if m.contains("maximum dimensions")),
        "unexpected error: {err:?}"
    );
}

/// Two unlimited dimensions mean libhdf5 would have chosen a v2 B-tree; an
/// extensible array claiming otherwise is malformed.
#[test]
fn ea_two_unlimited_dims_is_rejected() {
    let spec = EaSpec::default();
    let (buf, addr) = spec.build(&dense_elems(4));
    let chunk = [2u64, 2];
    let dataset = [4u64, 4];
    let max = [u64::MAX, u64::MAX];
    let err = parse_extensible_array(
        &buf,
        addr,
        &EaGeometry {
            chunk_dims: &chunk,
            dataset_dims: &dataset,
            max_dims: Some(&max),
            chunk_bytes: 16,
        },
    )
    .expect_err("must reject");
    assert!(
        matches!(err, OxiH5Error::Format(ref m) if m.contains("unlimited")),
        "unexpected error: {err:?}"
    );
}

/// A zero chunk dimension would divide by zero when scaling coordinates.
#[test]
fn ea_zero_chunk_dimension_is_rejected() {
    let spec = EaSpec::default();
    let (buf, addr) = spec.build(&dense_elems(4));
    let err = parse_extensible_array(&buf, addr, &geom_1d(&[0], &[16])).expect_err("must reject");
    assert!(
        matches!(err, OxiH5Error::Format(ref m) if m.contains("chunk dimension")),
        "unexpected error: {err:?}"
    );
}

/// Chunks left behind by a shrink are outside the current shape and must not be
/// handed to the readers, which size their output from the current dimensions.
#[test]
fn ea_chunks_beyond_current_shape_are_dropped() {
    let spec = EaSpec::default();
    let elems = dense_elems(20);
    let (buf, addr) = spec.build(&elems);
    // 20 chunks were allocated but the dataset now holds only 5 × 4 elements.
    let records = parse_extensible_array(&buf, addr, &geom_1d(&[4], &[20])).expect("shrunk");
    assert_eq!(records.len(), 5);
    assert_eq!(
        records.iter().map(|r| r.offsets[0]).collect::<Vec<_>>(),
        vec![0, 4, 8, 12, 16]
    );
}
