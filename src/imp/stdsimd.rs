#[cfg(feature = "nightly")]
mod imp {
    use core::{
        simd::{Simd, num::SimdUint as _},
        slice::ChunksExact,
    };

    const LANES: usize = 64;

    #[expect(clippy::decimal_literal_representation, reason = "Readability.")]
    const MOD: u32 = 65521;
    const NMAX: usize = 5552 & (!(LANES - 1));
    const WEIGHTS: Simd<u16, LANES> = {
        let mut weights = [0; LANES];
        let mut i = 0;
        #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
        #[expect(clippy::indexing_slicing, reason = "Const, cannot panic.")]
        while i < LANES {
            weights[LANES - 1 - i] = i as u16 + 1;
            i += 1;
        }

        Simd::from_array(weights)
    };

    // Compile-time proofs that operations cannot overflow
    #[expect(
        clippy::integer_division,
        clippy::integer_division_remainder_used,
        reason = "Intended."
    )]
    const _: () = {
        #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
        const MAX_A_GROWTH: u32 = (u8::MAX as u32) * (NMAX as u32);
        const MAX_A_INITIAL: u32 = MOD - 1;
        const MAX_A_BEFORE_MOD: u32 = MAX_A_INITIAL + MAX_A_GROWTH;
        const MAX_BYTE: u32 = u8::MAX as u32;
        const MAX_MUL_RESULT: u64 = (MAX_A_BEFORE_MOD as u64) * (LANES as u64);
        #[expect(clippy::cast_possible_truncation, reason = "Not possible.")]
        const MAX_PRODUCT: u32 = MAX_BYTE * (LANES as u32);
        #[expect(clippy::cast_possible_truncation, reason = "Not possible.")]
        const WEIGHTED_SUM: u32 = (NMAX * (NMAX + 1) / 2) as u32;

        assert!(
            (NMAX * (NMAX + 1) / 2) <= (u32::MAX as usize),
            "Weighted sum calculation has become unsafe."
        );

        assert!(
            LANES <= (u32::MAX as usize),
            "LANES was changed to an unsafe value."
        );

        assert!(
            MAX_PRODUCT <= (u16::MAX as u32),
            "Could not prove that `v_ * WEIGHTS` cannot overflow u16"
        );

        assert!(
            MAX_MUL_RESULT < (u32::MAX as u64),
            "Could not prove that `a * LANES` cannot overflow u32 when processing up to NMAX bytes"
        );

        assert!(
            MAX_A_BEFORE_MOD < u32::MAX,
            "Could not prove that accumulating NMAX bytes into `a` cannot overflow u32"
        );

        assert!(
            ((MAX_A_INITIAL as usize) * NMAX + (u8::MAX as usize) * (WEIGHTED_SUM as usize))
                < (u32::MAX as usize),
            "Could not prove b accumulation is safe; b grows by `a` contributions + weighted sum \
             of all bytes"
        );

        assert!(
            (MAX_A_INITIAL as usize + (LANES - 1) * (u8::MAX as usize)) < (u32::MAX as usize),
            "Could not prove that scalar remainder processing cannot overflow `a`"
        );
    };

    pub fn update(a: u16, b: u16, data: &[u8]) -> (u16, u16) {
        let mut a_ = u32::from(a);
        let mut b_ = u32::from(b);

        let chunks = data.chunks_exact(NMAX);
        let remainder = chunks.remainder();

        for chunk in chunks {
            update_simd(
                &mut a_,
                &mut b_,
                chunk.chunks_exact(LANES),
            );
            a_ %= MOD;
            b_ %= MOD;
        }

        let vs = remainder.chunks_exact(LANES);
        let vremainder = vs.remainder();
        update_simd(&mut a_, &mut b_, vs);

        for byte in vremainder {
            unsafe {
                a_ = a_.unchecked_add(u32::from(*byte));
                b_ = b_.unchecked_add(a_);
            }
        }

        a_ %= MOD;
        b_ %= MOD;

        #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
        (a_ as u16, b_ as u16)
    }

    fn update_simd(a_out: &mut u32, b_out: &mut u32, values: ChunksExact<u8>) {
        let (mut a, mut b) = (*a_out, *b_out);

        #[expect(clippy::arithmetic_side_effects, reason = "Intended.")]
        #[expect(clippy::cast_possible_truncation, reason = "Special std::simd op.")]
        for v in values {
            let v_ = Simd::from_slice(v).cast::<u16>();
            unsafe {
                b = b
                    .unchecked_add(a.unchecked_mul(LANES as u32))
                    .unchecked_add(
                        (v_ * WEIGHTS)
                            .cast::<u32>()
                            .reduce_sum(),
                    );
                a = a.unchecked_add(u32::from(v_.reduce_sum()));
            }
        }

        *a_out = a;
        *b_out = b;
    }
}

use super::Adler32Imp;

#[must_use]
pub fn get_imp() -> Option<Adler32Imp> { get_imp_inner() }

#[inline]
#[cfg(feature = "nightly")]
#[expect(clippy::unnecessary_wraps, reason = "Match API.")]
fn get_imp_inner() -> Option<Adler32Imp> { Some(imp::update) }

#[inline]
#[cfg(not(feature = "nightly"))]
fn get_imp_inner() -> Option<Adler32Imp> { None }

#[cfg(test)]
mod tests {
    use rand::Rng;

    #[test]
    fn zeroes() {
        assert_sum_eq(&[]);
        assert_sum_eq(&[0]);
        assert_sum_eq(&[0, 0]);
        assert_sum_eq(&[0; 100]);
        assert_sum_eq(&[0; 1024]);
        assert_sum_eq(&[0; 1024 * 1024]);
    }

    #[test]
    fn ones() {
        assert_sum_eq(&[]);
        assert_sum_eq(&[1]);
        assert_sum_eq(&[1, 1]);
        assert_sum_eq(&[1; 100]);
        assert_sum_eq(&[1; 1024]);
        assert_sum_eq(&[1; 1024 * 1024]);
    }

    #[test]
    fn random() {
        let mut random = [0; 1024 * 1024];
        rand::rng().fill(&mut random[..]);

        assert_sum_eq(&random[..1]);
        assert_sum_eq(&random[..8]);
        assert_sum_eq(&random[..64]);
        assert_sum_eq(&random[..100]);
        assert_sum_eq(&random[..1024]);
        assert_sum_eq(&random[..1024 * 1024]);
    }

    /// Example calculation from https://en.wikipedia.org/wiki/Adler-32.
    #[test]
    fn wiki() { assert_sum_eq(b"Wikipedia"); }

    fn assert_sum_eq(data: &[u8]) {
        if let Some(update) = super::get_imp() {
            let (a, b) = update(1, 0, data);
            let left = u32::from(b) << 16 | u32::from(a);
            let right = adler::adler32_slice(data);

            assert_eq!(left, right, "len({})", data.len());
        }
    }
}
