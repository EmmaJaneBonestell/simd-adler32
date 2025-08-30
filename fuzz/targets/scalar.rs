#![no_main]
#[macro_use]
extern crate libfuzzer_sys;

fuzz_target!(|data: &[u8]| {
    let _ = simd_adler32::imp::scalar::update(1, 0, data);
});
