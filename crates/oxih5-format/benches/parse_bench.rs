use criterion::{criterion_group, criterion_main, Criterion};
use oxih5_format::{header, superblock};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// HDF5 file signature (matches superblock.rs constant)
// ---------------------------------------------------------------------------

const HDF5_SIG: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

// ---------------------------------------------------------------------------
// Superblock fixtures
// ---------------------------------------------------------------------------

fn build_superblock_v0() -> Vec<u8> {
    let mut sb = vec![0u8; 96];
    sb[0..8].copy_from_slice(&HDF5_SIG);
    sb[8] = 0;
    sb[13] = 8;
    sb[14] = 8;
    sb[64..72].copy_from_slice(&96_u64.to_le_bytes());
    sb
}

fn build_superblock_v2() -> Vec<u8> {
    let soo: u8 = 8;
    let total = 12 + 4 * soo as usize + 4;
    let mut sb = vec![0u8; total];
    sb[0..8].copy_from_slice(&HDF5_SIG);
    sb[8] = 2;
    sb[9] = soo;
    sb[10] = soo;
    sb[11] = 0;
    sb[20..28].copy_from_slice(&u64::MAX.to_le_bytes());
    sb[28..36].copy_from_slice(&1024_u64.to_le_bytes());
    sb[36..44].copy_from_slice(&48_u64.to_le_bytes());
    sb
}

fn build_superblock_v3() -> Vec<u8> {
    let soo: u8 = 8;
    let total = 12 + 4 * soo as usize + 4;
    let mut sb = vec![0u8; total];
    sb[0..8].copy_from_slice(&HDF5_SIG);
    sb[8] = 3;
    sb[9] = soo;
    sb[10] = soo;
    sb[36..44].copy_from_slice(&100_u64.to_le_bytes());
    sb
}

// ---------------------------------------------------------------------------
// Header fixtures
// ---------------------------------------------------------------------------

/// Minimal v1 object header with one dataspace message (1 message total).
fn build_v1_1msg() -> Vec<u8> {
    let mut data = vec![0u8; 256];
    data[0] = 1;
    data[2] = 1;
    data[4] = 1;
    data[8..12].copy_from_slice(&16u32.to_le_bytes());
    data[16] = 0x01;
    data[17] = 0x00;
    data[18] = 0x08;
    data[19] = 0x00;
    data[20] = 0x00;
    data[24..32].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22]);
    data
}

/// v1 object header with 64 messages of type 0x01, each with an 8-byte body.
fn build_v1_64msg() -> Vec<u8> {
    let msg_count: u32 = 64;
    // Each message: 8-byte header + 8-byte body (already 8-byte aligned) = 16 bytes.
    let header_data_size: u32 = msg_count * 16;
    let total = 16 + header_data_size as usize + 64;
    let mut data = vec![0u8; total];
    data[0] = 1;
    data[2..4].copy_from_slice(&(msg_count as u16).to_le_bytes());
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    data[8..12].copy_from_slice(&header_data_size.to_le_bytes());
    let mut pos = 16usize;
    for i in 0..msg_count as usize {
        data[pos] = 0x01;
        data[pos + 1] = 0x00;
        data[pos + 2] = 0x08;
        data[pos + 3] = 0x00;
        data[pos + 4] = 0x00;
        let val = i as u64;
        data[pos + 8..pos + 16].copy_from_slice(&val.to_le_bytes());
        pos += 16;
    }
    data
}

/// Minimal v2 OHDR with one message (flags=0, chunk_size_size=1 byte).
fn build_v2_1msg() -> Vec<u8> {
    let chunk0_size: usize = 12;
    let total = 4 + 1 + 1 + 1 + chunk0_size + 4 + 16;
    let mut data = vec![0u8; total];
    data[0..4].copy_from_slice(b"OHDR");
    data[4] = 2;
    data[5] = 0;
    data[6] = chunk0_size as u8;
    let p = 7;
    data[p] = 0x01;
    data[p + 1] = 0x04;
    data[p + 2] = 0x00;
    data[p + 3] = 0x00;
    data[p + 4] = 0xAA;
    data[p + 5] = 0xBB;
    data[p + 6] = 0xCC;
    data[p + 7] = 0xDD;
    data
}

/// v2 OHDR with 64 messages (flags=0x01: chunk_size_size=2 bytes to fit larger chunk).
fn build_v2_64msg() -> Vec<u8> {
    let msg_count: usize = 64;
    // Each v2 message: 4-byte header + 8-byte body = 12 bytes. 64 * 12 = 768.
    let per_msg = 12usize;
    let chunk0_size: u16 = (msg_count * per_msg + 4) as u16; // +4 for NIL
                                                             // flags=0x01: bits 0-1 = 1 → chunk_size_size = 2 bytes
    let total = 4 + 1 + 1 + 2 + chunk0_size as usize + 4 + 64;
    let mut data = vec![0u8; total];
    data[0..4].copy_from_slice(b"OHDR");
    data[4] = 2;
    data[5] = 0x01;
    data[6..8].copy_from_slice(&chunk0_size.to_le_bytes());
    let mut pos = 8usize;
    for i in 0..msg_count {
        data[pos] = 0x01;
        let body_size: u16 = 8;
        data[pos + 1..pos + 3].copy_from_slice(&body_size.to_le_bytes());
        data[pos + 3] = 0x00;
        let val = i as u64;
        data[pos + 4..pos + 12].copy_from_slice(&val.to_le_bytes());
        pos += per_msg;
    }
    // NIL terminator already zero
    data
}

// ---------------------------------------------------------------------------
// Bench groups
// ---------------------------------------------------------------------------

fn bench_superblock(c: &mut Criterion) {
    let v0 = build_superblock_v0();
    let v2 = build_superblock_v2();
    let v3 = build_superblock_v3();

    let mut grp = c.benchmark_group("bench_superblock");
    grp.bench_function("parse_v0", |b| {
        b.iter(|| superblock::parse(black_box(&v0)).unwrap())
    });
    grp.bench_function("parse_v2", |b| {
        b.iter(|| superblock::parse(black_box(&v2)).unwrap())
    });
    grp.bench_function("parse_v3", |b| {
        b.iter(|| superblock::parse(black_box(&v3)).unwrap())
    });
    grp.finish();
}

fn bench_header(c: &mut Criterion) {
    let v1_1 = build_v1_1msg();
    let v1_64 = build_v1_64msg();
    let v2_1 = build_v2_1msg();
    let v2_64 = build_v2_64msg();

    let mut grp = c.benchmark_group("bench_header");
    grp.bench_function("parse_v1_1msg", |b| {
        b.iter(|| header::parse_messages(black_box(&v1_1), 0).unwrap())
    });
    grp.bench_function("parse_v1_64msg", |b| {
        b.iter(|| header::parse_messages(black_box(&v1_64), 0).unwrap())
    });
    grp.bench_function("parse_v2_1msg", |b| {
        b.iter(|| header::parse_messages(black_box(&v2_1), 0).unwrap())
    });
    grp.bench_function("parse_v2_64msg", |b| {
        b.iter(|| header::parse_messages(black_box(&v2_64), 0).unwrap())
    });
    grp.finish();
}

criterion_group!(benches, bench_superblock, bench_header);
criterion_main!(benches);
