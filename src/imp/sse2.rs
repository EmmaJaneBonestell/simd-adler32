#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "std", target_feature = "sse2")
))]
mod imp {
    #[expect(clippy::decimal_literal_representation, reason = "Readability.")]
    const MOD: u32 = 65521;
    const NMAX: usize = 5552;
    const BLOCK_SIZE: usize = 32;
    #[expect(
        clippy::integer_division,
        clippy::integer_division_remainder_used,
        reason = "Intended."
    )]
    const MAX_BLOCKS: usize = NMAX / BLOCK_SIZE;
    const CHUNK_SIZE: usize = MAX_BLOCKS * BLOCK_SIZE;

    // Compile-time proofs that operations cannot overflow
    const _: () = {
        const MAX_A: u32 = MOD - 1;
        assert!(
            (NMAX * (u8::MAX as usize)) < (u32::MAX as usize),
            "Prove that accumulating NMAX bytes of u8::MAX cannot overflow"
        );
        assert!(
            ((MAX_A as usize) * MAX_BLOCKS) < (u32::MAX as usize),
            "Prove that a * blocks.len() cannot overflow"
        );
        assert!(
            ((MOD as usize - 1) * NMAX) < (u32::MAX as usize),
            "Prove b accumulation is safe; b grows at most `a * NMAX` per chunk"
        );
        // Slightly redundant
        assert!(
            (MAX_BLOCKS * BLOCK_SIZE * 255) < (u32::MAX as usize),
            "Prove that reduce_add result cannot overflow when added to a"
        );
    };

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_madd_epi16, _mm_sad_epu8,
        _mm_set_epi8, _mm_set_epi32, _mm_setzero_si128, _mm_shuffle_epi32, _mm_slli_epi32,
        _mm_unpackhi_epi8, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_madd_epi16, _mm_sad_epu8,
        _mm_set_epi8, _mm_set_epi32, _mm_setzero_si128, _mm_shuffle_epi32, _mm_slli_epi32,
        _mm_unpackhi_epi8, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    };

    use crate::imp::_MM_SHUFFLE;

    pub fn update(a: u16, b: u16, data: &[u8]) -> (u16, u16) { unsafe { update_imp(a, b, data) } }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn update_imp(a: u16, b: u16, data: &[u8]) -> (u16, u16) {
        let mut a_ = u32::from(a);
        let mut b_ = u32::from(b);

        let chunks = data.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();
        for chunk in chunks {
            unsafe {
                update_chunk_block(&mut a_, &mut b_, chunk);
            }
        }

        unsafe {
            update_block(&mut a_, &mut b_, remainder);
        };

        #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
        (a_ as u16, b_ as u16)
    }

    unsafe fn update_chunk_block(a: &mut u32, b: &mut u32, chunk: &[u8]) {
        debug_assert_eq!(
            chunk.len(),
            CHUNK_SIZE,
            "Unexpected chunk size (expected {}, got {})",
            CHUNK_SIZE,
            chunk.len()
        );

        unsafe {
            reduce_add_blocks(a, b, chunk);
        };

        *a %= MOD;
        *b %= MOD;
    }

    unsafe fn update_block(a: &mut u32, b: &mut u32, chunk: &[u8]) {
        debug_assert!(
            chunk.len() <= CHUNK_SIZE,
            "Unexpected chunk size (expected <= {}, got {})",
            CHUNK_SIZE,
            chunk.len()
        );

        unsafe {
            for byte in reduce_add_blocks(a, b, chunk) {
                *a = a.unchecked_add(u32::from(*byte));
                *b = b.unchecked_add(*a);
            }
        }

        *a %= MOD;
        *b %= MOD;
    }

    #[inline(always)]
    unsafe fn reduce_add_blocks<'a>(a: &mut u32, b: &mut u32, chunk: &'a [u8]) -> &'a [u8] {
        unsafe {
            if chunk.len() < BLOCK_SIZE {
                return chunk;
            }

            let blocks = chunk.chunks_exact(BLOCK_SIZE);
            let blocks_remainder = blocks.remainder();

            let zero_v = _mm_setzero_si128();
            let weight_hi_v = get_weight_hi();
            let weight_lo_v = get_weight_lo();

            #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
            #[expect(
                clippy::cast_possible_wrap,
                reason = "Does not happen on valid inputs; asserted for."
            )]
            let mut p_v = _mm_set_epi32(
                0,
                0,
                0,
                (a.unchecked_mul(blocks.len() as u32)) as i32,
            );
            let mut a_v = _mm_setzero_si128();
            #[expect(
                clippy::cast_possible_wrap,
                reason = "Does not happen on valid inputs; asserted for."
            )]
            let mut b_v = _mm_set_epi32(0, 0, 0, *b as i32);

            for block in blocks {
                #[expect(clippy::cast_ptr_alignment, reason = "Unaligned reads used.")]
                let block_ptr = block.as_ptr().cast::<__m128i>();
                let left_v = _mm_loadu_si128(block_ptr);
                let right_v = _mm_loadu_si128(block_ptr.add(1));

                p_v = _mm_add_epi32(p_v, a_v);

                a_v = _mm_add_epi32(a_v, _mm_sad_epu8(left_v, zero_v));
                let mad = maddubs(left_v, weight_hi_v);
                b_v = _mm_add_epi32(b_v, mad);

                a_v = _mm_add_epi32(a_v, _mm_sad_epu8(right_v, zero_v));
                let mad1 = maddubs(right_v, weight_lo_v);
                b_v = _mm_add_epi32(b_v, mad1);
            }

            b_v = _mm_add_epi32(b_v, _mm_slli_epi32(p_v, 5));

            *a = a.unchecked_add(reduce_add(a_v));
            *b = reduce_add(b_v);

            blocks_remainder
        }
    }

    #[inline(always)]
    unsafe fn maddubs(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let a_lo = _mm_unpacklo_epi8(a, _mm_setzero_si128());
            let a_hi = _mm_unpackhi_epi8(a, _mm_setzero_si128());

            let b_lo = _mm_unpacklo_epi8(b, _mm_setzero_si128());
            let b_hi = _mm_unpackhi_epi8(b, _mm_setzero_si128());

            let lo = _mm_madd_epi16(a_lo, b_lo);
            let hi = _mm_madd_epi16(a_hi, b_hi);

            _mm_add_epi32(lo, hi)
        }
    }

    #[expect(clippy::cast_sign_loss, reason = "Intended.")]
    #[inline(always)]
    unsafe fn reduce_add(v: __m128i) -> u32 {
        unsafe {
            let hi = _mm_unpackhi_epi64(v, v);
            let sum = _mm_add_epi32(hi, v);
            #[expect(clippy::used_underscore_items, reason = "Existing name.")]
            let hi1 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1));

            let sum1 = _mm_add_epi32(sum, hi1);

            _mm_cvtsi128_si32(sum1) as u32
        }
    }

    #[inline(always)]
    unsafe fn get_weight_lo() -> __m128i {
        unsafe {
            _mm_set_epi8(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            )
        }
    }

    #[inline(always)]
    unsafe fn get_weight_hi() -> __m128i {
        unsafe {
            _mm_set_epi8(
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            )
        }
    }
}

use super::Adler32Imp;

/// Resolves update implementation if CPU supports sse2 instructions.
#[must_use]
pub fn get_imp() -> Option<Adler32Imp> { get_imp_inner() }

#[inline]
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn get_imp_inner() -> Option<Adler32Imp> {
    std::arch::is_x86_feature_detected!("sse2").then_some(imp::update as Adler32Imp)
}

#[inline]
#[cfg(all(
    target_feature = "sse2",
    not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))
))]
fn get_imp_inner() -> Option<Adler32Imp> { Some(imp::update) }

#[inline]
#[cfg(all(
    not(target_feature = "sse2"),
    not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))
))]
fn get_imp_inner() -> Option<Adler32Imp> { None }

#[cfg(test)]
mod tests {
    use rand::Rng as _;

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
        assert_sum_eq(&random[..100]);
        assert_sum_eq(&random[..1024]);
        assert_sum_eq(&random[..1024 * 1024]);
    }

    /// Example calculation from <https://en.wikipedia.org/wiki/Adler-32>.
    #[test]
    fn wiki() { assert_sum_eq(b"Wikipedia"); }

    fn assert_sum_eq(data: &[u8]) {
        if let Some(update) = super::get_imp() {
            let (a, b) = update(1, 0, data);
            let left = (u32::from(b) << 16_i32) | u32::from(a);
            let right = adler::adler32_slice(data);

            assert_eq!(left, right, "len({})", data.len());
        }
    }
}
