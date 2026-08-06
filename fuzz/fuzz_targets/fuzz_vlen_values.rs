#![no_main]
use libfuzzer_sys::fuzz_target;
use oxih5_core::{ByteOrder, Dtype};

// Drives the variable-length / global-heap value decoders in `values.rs`
// directly. `n_elems`, the vlen sequence length and the global-heap object
// index are all attacker-controlled integers read straight off disk in the
// real parse path, and `decode_vlen_strings` / `decode_vlen_sequences` /
// `decode_object_refs` each multiply one of them by a fixed element width to
// size a buffer — precisely the pattern that needed a `checked_mul` fix for
// 32-bit/wasm32 `usize` overflow. Fuzzing the checked path here guards
// against a future edit reintroducing an unchecked multiply.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // Keep n_elems small in magnitude so the fuzzer explores the "just above
    // / just below the buffer length" boundary rather than only ever hitting
    // the overflow-rejection path.
    let n_elems = data[0] as usize;
    let body = &data[1..];

    // `decode_vlen_strings` dereferences each 16-byte reference in `body`
    // through the global heap in `body` itself (a real file's vlen
    // references and the global-heap collections they point into both live
    // in the same byte stream, so reusing `body` for both parameters is
    // representative, not a shortcut).
    let _ = oxih5_format::values::decode_vlen_strings(body, body, n_elems);
    let _ = oxih5_format::values::decode_object_refs(body, n_elems);

    let base = Dtype::Int {
        size: 4,
        signed: true,
        order: ByteOrder::Little,
    };
    // `decode_vlen_sequences` additionally dereferences a global-heap
    // reference against `body` as the whole file, so pass it as both the
    // "file" and the "vlen reference array" — the parser bounds-checks every
    // read against `body`'s actual length regardless.
    let _ = oxih5_format::values::decode_vlen_sequences(body, body, n_elems, &base);
});
