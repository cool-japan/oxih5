// traverse_bench.rs — Traversal allocation-cost benchmark for OxiH5.
//
// FileWriter caps at 8 flat datasets with no nested groups (write.rs:248-252),
// so a genuine >1GB many-group file cannot be constructed via the public writer.
// This bench measures the traversal allocation cost using a synthetic
// object-count proxy: hand-crafted B-tree / SNOD / local-heap in-memory
// buffers with varying entry counts.
//
// Optimizations measured:
//   T6: Vec::with_capacity pre-sizing in group traversal loops
//   T7: LocalHeap borrows &'a [u8] — no copy of the data segment
//   T8: FractalHeap takes &[u8] — no Arc::new(file_data.to_vec()) per group

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxih5_format::{heap, snod};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// HDF5 bytes-level constants
// ---------------------------------------------------------------------------

/// Write a u16 LE at `off` in `buf`.
fn w16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

/// Write a u64 LE at `off` in `buf`.
fn w64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Build a flat HDF5 in-memory image containing:
///   - A superblock-v0 stub (minimal, enough for `heap::parse` and `snod::parse`)
///   - One local heap with `num_entries` NUL-terminated ASCII names
///   - One SNOD node referencing those names
///   - One TREE (B-tree v1 level-0) pointing at the single SNOD
///
/// Layout:
///   [0..96]     superblock (stub, unused by parse functions called here)
///   [96..128]   local heap header (32 bytes)
///   [128..128+heap_seg_size]  heap data segment (packed names)
///   [...] SNOD node
///   [...] TREE node
struct Fixture {
    data: Vec<u8>,
    btree_addr: u64,
    heap_addr: u64,
}

/// Build a name like "dataset_NNN\0" into the heap data segment and SNOD.
fn build_fixture(num_entries: usize) -> Fixture {
    // -----------------------------------------------------------------------
    // Step 1: build heap data segment (packed NUL-terminated names)
    // -----------------------------------------------------------------------

    // Each name: "dataset_NNN\0" = at most 16 bytes (for NNN < 1000).
    let max_name_len: usize = 16;
    let heap_seg_size: usize = (num_entries * max_name_len).max(64);

    let mut heap_segment = vec![0u8; heap_seg_size];
    let mut name_offsets: Vec<u64> = Vec::with_capacity(num_entries);

    let mut cursor: usize = 0;
    for i in 0..num_entries {
        let name = format!("dataset_{i:03}\0");
        let bytes = name.as_bytes();
        heap_segment[cursor..cursor + bytes.len()].copy_from_slice(bytes);
        name_offsets.push(cursor as u64);
        cursor += bytes.len();
    }

    // -----------------------------------------------------------------------
    // Step 2: plan memory layout
    // -----------------------------------------------------------------------

    // heap header is 32 bytes; placed at base 96 (after stub superblock).
    let heap_hdr_addr: usize = 96;
    let heap_data_addr: usize = heap_hdr_addr + 32;
    let heap_data_end: usize = heap_data_addr + heap_seg_size;

    // SNOD node: 8-byte header + num_entries * 40 bytes per entry.
    let snod_addr: usize = heap_data_end;
    let snod_entry_count: u16 = num_entries as u16;
    let snod_size: usize = 8 + (num_entries * 40);
    let snod_end: usize = snod_addr + snod_size;

    // B-tree v1 level-0 node: 24-byte fixed header + (K+1)*8 keys + K*8 children.
    // We have exactly 1 child (the SNOD) → K=1 → 2 keys + 1 child = 3*8=24 bytes keys/children.
    // Layout: sig(4) + type(1) + level(1) + K(2) + left_sib(8) + right_sib(8) +
    //         key[0](8) + child[0](8) + key[1](8) = 48 bytes total.
    let btree_addr: usize = snod_end;
    let btree_size: usize = 48;
    let total_size: usize = btree_addr + btree_size;

    // -----------------------------------------------------------------------
    // Step 3: assemble buffer
    // -----------------------------------------------------------------------

    let mut buf = vec![0u8; total_size];

    // Heap header at heap_hdr_addr:
    //   [0..4]   "HEAP"
    //   [4]      version = 0
    //   [5..8]   reserved
    //   [8..16]  data_segment_size (u64 LE)
    //   [16..24] free_list_head    (u64 LE)
    //   [24..32] data_segment_addr (u64 LE)
    let h = heap_hdr_addr;
    buf[h..h + 4].copy_from_slice(b"HEAP");
    buf[h + 4] = 0; // version
    w64(&mut buf, h + 8, heap_seg_size as u64);
    w64(&mut buf, h + 16, heap_seg_size as u64); // free list = end of segment
    w64(&mut buf, h + 24, heap_data_addr as u64);

    // Copy heap data segment.
    buf[heap_data_addr..heap_data_addr + heap_seg_size].copy_from_slice(&heap_segment);

    // SNOD at snod_addr:
    //   [0..4]  "SNOD"
    //   [4]     version = 1
    //   [5]     reserved = 0
    //   [6..8]  num_symbols (u16 LE)
    //   [8..]   entries (40 bytes each)
    //
    // Each entry (40 bytes, soo=8):
    //   [0..8]   name_offset (u64 LE, into heap data segment)
    //   [8..16]  object_header_address (u64 LE)
    //   [16..20] cache_type (u32 LE) = 0
    //   [20..24] reserved (u32 LE)
    //   [24..40] scratch (16 bytes, zero)
    let s = snod_addr;
    buf[s..s + 4].copy_from_slice(b"SNOD");
    buf[s + 4] = 1; // version
    w16(&mut buf, s + 6, snod_entry_count);
    for (i, &name_off) in name_offsets.iter().enumerate() {
        let e = s + 8 + i * 40;
        w64(&mut buf, e, name_off);
        // object_header_address: use a dummy non-MAX address.
        w64(&mut buf, e + 8, (btree_addr as u64) + 1000 + i as u64);
        // cache_type, reserved, scratch: all zero (already zero-initialised).
    }

    // B-tree v1 level-0 node at btree_addr:
    //   [0..4]   "TREE"
    //   [4]      node_type = 0 (group)
    //   [5]      level = 0 (leaf)
    //   [6..8]   entries_used K = 1
    //   [8..16]  left_sibling  = u64::MAX (undefined)
    //   [16..24] right_sibling = u64::MAX
    //   [24..32] key[0] (8 bytes)
    //   [32..40] child[0] = snod_addr
    //   [40..48] key[1] (8 bytes, terminator)
    let t = btree_addr;
    buf[t..t + 4].copy_from_slice(b"TREE");
    buf[t + 4] = 0; // node_type: group
    buf[t + 5] = 0; // level: leaf
    w16(&mut buf, t + 6, 1); // entries_used = 1
    w64(&mut buf, t + 8, u64::MAX); // left sibling = undefined
    w64(&mut buf, t + 16, u64::MAX); // right sibling = undefined
                                     // key[0] at t+24: zero (key data, ignored by group traverse)
    w64(&mut buf, t + 32, snod_addr as u64); // child[0] = SNOD address
                                             // key[1] at t+40: zero terminator

    Fixture {
        data: buf,
        btree_addr: btree_addr as u64,
        heap_addr: heap_hdr_addr as u64,
    }
}

// ---------------------------------------------------------------------------
// Bench: heap::parse + name_at (T7 — LocalHeap borrow)
// ---------------------------------------------------------------------------

/// Measure the cost of `heap::parse` + `name_at` for `num_entries` names.
/// Before T7 this paid a `to_vec()` copy of the entire heap data segment.
/// After T7 `LocalHeap` borrows the slice — no copy.
fn bench_heap_parse(c: &mut Criterion) {
    let mut grp = c.benchmark_group("heap_parse_name_at");

    for &n in &[8usize, 32, 64, 128] {
        let fix = build_fixture(n);

        grp.bench_with_input(BenchmarkId::new("entries", n), &n, |b, _| {
            b.iter(|| {
                let lh = heap::parse(black_box(&fix.data), black_box(fix.heap_addr))
                    .expect("heap::parse");
                // Enumerate all names to exercise name_at fully.
                // Each name slot is 16 bytes wide (max_name_len from build_fixture).
                for off in 0..n {
                    let name_off = off * 16; // usize * usize = usize, no cast needed
                    if name_off < lh.data.len() {
                        let _ = lh.name_at(black_box(name_off));
                    }
                }
            })
        });
    }

    grp.finish();
}

// ---------------------------------------------------------------------------
// Bench: snod::parse (object-count proxy for traversal)
// ---------------------------------------------------------------------------

/// Measure the cost of `snod::parse` over a SNOD with varying entry counts.
/// This is the inner loop of old-style B-tree group traversal.
fn bench_snod_parse(c: &mut Criterion) {
    let mut grp = c.benchmark_group("snod_parse");

    for &n in &[8usize, 32, 64, 128] {
        let fix = build_fixture(n);
        // Locate SNOD in the fixture: after the heap header (32) + heap segment.
        let heap_seg_size = (n * 16).max(64);
        let snod_offset = (96 + 32 + heap_seg_size) as u64;

        grp.bench_with_input(BenchmarkId::new("entries", n), &n, |b, _| {
            b.iter(|| {
                snod::parse(black_box(&fix.data), black_box(snod_offset)).expect("snod::parse")
            })
        });
    }

    grp.finish();
}

// ---------------------------------------------------------------------------
// Bench: full old-style group traversal path (heap + btree + snod + name_at)
// ---------------------------------------------------------------------------

/// Exercise the complete old-style group traversal: `heap::parse` (T7) +
/// `btree::parse` + `snod::parse` + `name_at`.
///
/// This is the inner loop hit for every group in a large HDF5 file with
/// many old-style groups.  The T7 optimisation eliminates one `to_vec()`
/// allocation per call; T6 eliminates one `Vec` reallocation per traversal.
fn bench_group_traverse(c: &mut Criterion) {
    let mut grp = c.benchmark_group("group_traverse_btree");

    for &n in &[8usize, 32, 64, 128] {
        let fix = build_fixture(n);

        grp.bench_with_input(BenchmarkId::new("entries", n), &n, |b, _| {
            b.iter(|| {
                // Mirrors the hot path in group::list_datasets.
                let lh = heap::parse(black_box(&fix.data), black_box(fix.heap_addr))
                    .expect("heap::parse");
                let tree =
                    oxih5_format::btree::parse(black_box(&fix.data), black_box(fix.btree_addr))
                        .expect("btree::parse");
                // T6: pre-sized Vec.
                let mut names = Vec::with_capacity(tree.leaf_addresses.len() * 8);
                for &leaf_addr in &tree.leaf_addresses {
                    let entries = snod::parse(black_box(&fix.data), black_box(leaf_addr))
                        .expect("snod::parse");
                    for entry in entries {
                        let name = lh.name_at(entry.name_offset as usize).expect("name_at");
                        if !name.is_empty() {
                            names.push(name.to_string());
                        }
                    }
                }
                black_box(names)
            })
        });
    }

    grp.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(
    traverse_benches,
    bench_heap_parse,
    bench_snod_parse,
    bench_group_traverse,
);
criterion_main!(traverse_benches);
