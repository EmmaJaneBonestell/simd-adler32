#[cfg(all(
    feature = "nightly",
    any(all(target_arch = "arm", target_feature = "v7",), target_arch = "aarch64"),
    any(feature = "std", target_feature = "neon")
))]
mod imp {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::{
        uint8x16_t, uint32x4_t, vaddq_u32, vaddvq_u32, vdotq_u32, vpaddlq_u8, vpaddlq_u16,
        vshlq_n_u32,
    };
    #[cfg(target_arch = "arm")]
    use core::arch::arm::{
        uint8x16_t, vaddw_u16, vdupq_n_u32, vget_high_u8, vget_high_u16, vget_low_u8, vget_low_u16,
        vld1q_u8, vmlal_u16, vmovl_u8,
    };

    #[expect(clippy::decimal_literal_representation, reason = "Readability.")]
    const MOD: u32 = 65521;
    const NMAX: usize = 5552;
    const BLOCK_SIZE: usize = 64;
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

    pub fn update(a: u16, b: u16, data: &[u8]) -> (u16, u16) { unsafe { update_imp(a, b, data) } }

    #[inline]
    #[cfg_attr(target_arch = "arm", target_feature(enable = "v7,neon"))]
    #[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon,dotprod"))]
    unsafe fn update_imp(a: u16, b: u16, data: &[u8]) -> (u16, u16) {
        let mut a_ = u32::from(a);
        let mut b_ = u32::from(b);

        let chunks = data.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();
        for chunk in chunks {
            unsafe {
                update_chunk_block(&mut a_, &mut b_, chunk);
            };
        }

        unsafe {
            update_block(&mut a_, &mut b_, remainder);
        };

        #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
        (a_ as u16, b_ as u16)
    }

    #[inline]
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

    #[inline]
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

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn reduce_add_blocks<'a>(a: &mut u32, b: &mut u32, chunk: &'a [u8]) -> &'a [u8] {
        unsafe {
            use core::mem::transmute;

            const UNROLL_SIZE: usize = BLOCK_SIZE * 2;

            if chunk.len() < UNROLL_SIZE {
                if chunk.len() < BLOCK_SIZE {
                    return chunk;
                }

                let blocks = chunk.chunks_exact(BLOCK_SIZE);
                let blocks_remainder = blocks.remainder();

                let weight_0 = get_weight_0();
                let weight_1 = get_weight_1();
                let weight_2 = get_weight_2();
                let weight_3 = get_weight_3();

                #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
                let mut p_v: uint32x4_t =
                    transmute([a.unchecked_mul(blocks.len() as u32), 0, 0, 0]);
                let mut a_v: uint32x4_t = transmute([0_u32, 0, 0, 0]);
                let mut b_v: uint32x4_t = transmute([*b, 0, 0, 0]);

                for block in blocks {
                    #[expect(clippy::cast_ptr_alignment, reason = "Unaligned reads used.")]
                    let block_ptr = block.as_ptr().cast::<uint8x16_t>();
                    let v0 = block_ptr.read_unaligned();
                    let v1 = block_ptr.add(1).read_unaligned();
                    let v2 = block_ptr.add(2).read_unaligned();
                    let v3 = block_ptr.add(3).read_unaligned();

                    p_v = vaddq_u32(p_v, a_v);

                    a_v = vaddq_u32(a_v, vpaddlq_u16(vpaddlq_u8(v0)));
                    b_v = vdotq_u32(b_v, v0, weight_0);

                    a_v = vaddq_u32(a_v, vpaddlq_u16(vpaddlq_u8(v1)));
                    b_v = vdotq_u32(b_v, v1, weight_1);

                    a_v = vaddq_u32(a_v, vpaddlq_u16(vpaddlq_u8(v2)));
                    b_v = vdotq_u32(b_v, v2, weight_2);

                    a_v = vaddq_u32(a_v, vpaddlq_u16(vpaddlq_u8(v3)));
                    b_v = vdotq_u32(b_v, v3, weight_3);
                }

                b_v = vaddq_u32(b_v, vshlq_n_u32(p_v, 6));

                *a = a.unchecked_add(vaddvq_u32(a_v));
                *b = vaddvq_u32(b_v);

                return blocks_remainder;
            }

            let blocks = chunk.chunks_exact(UNROLL_SIZE);
            let blocks_remainder = blocks.remainder();

            let weight_0 = get_weight_0_128();
            let weight_1 = get_weight_1_128();
            let weight_2 = get_weight_2_128();
            let weight_3 = get_weight_3_128();
            let weight_4 = get_weight_4_128();
            let weight_5 = get_weight_5_128();
            let weight_6 = get_weight_6_128();
            let weight_7 = get_weight_7_128();

            #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
            let mut p_v: uint32x4_t = transmute([a.unchecked_mul(blocks.len() as u32), 0, 0, 0]);
            let mut a0_v: uint32x4_t = transmute([0_u32, 0, 0, 0]);
            let mut a1_v: uint32x4_t = transmute([0_u32, 0, 0, 0]);
            let mut b0_v: uint32x4_t = transmute([*b, 0, 0, 0]);
            let mut b1_v: uint32x4_t = transmute([0_u32, 0, 0, 0]);

            for block in blocks {
                #[expect(clippy::cast_ptr_alignment, reason = "Unaligned reads used.")]
                let ptr1 = block.as_ptr().cast::<uint8x16_t>();
                let ptr2 = ptr1.add(4);

                let v0_1 = ptr1.read_unaligned();
                let v1_1 = ptr1.add(1).read_unaligned();
                let v2_1 = ptr1.add(2).read_unaligned();
                let v3_1 = ptr1.add(3).read_unaligned();

                let v0_2 = ptr2.read_unaligned();
                let v1_2 = ptr2.add(1).read_unaligned();
                let v2_2 = ptr2.add(2).read_unaligned();
                let v3_2 = ptr2.add(3).read_unaligned();

                p_v = vaddq_u32(p_v, vaddq_u32(a0_v, a1_v));

                a0_v = vaddq_u32(a0_v, vpaddlq_u16(vpaddlq_u8(v0_1)));
                b0_v = vdotq_u32(b0_v, v0_1, weight_0);
                a0_v = vaddq_u32(a0_v, vpaddlq_u16(vpaddlq_u8(v1_1)));
                b0_v = vdotq_u32(b0_v, v1_1, weight_1);
                a0_v = vaddq_u32(a0_v, vpaddlq_u16(vpaddlq_u8(v2_1)));
                b0_v = vdotq_u32(b0_v, v2_1, weight_2);
                a0_v = vaddq_u32(a0_v, vpaddlq_u16(vpaddlq_u8(v3_1)));
                b0_v = vdotq_u32(b0_v, v3_1, weight_3);

                a1_v = vaddq_u32(a1_v, vpaddlq_u16(vpaddlq_u8(v0_2)));
                b1_v = vdotq_u32(b1_v, v0_2, weight_4);
                a1_v = vaddq_u32(a1_v, vpaddlq_u16(vpaddlq_u8(v1_2)));
                b1_v = vdotq_u32(b1_v, v1_2, weight_5);
                a1_v = vaddq_u32(a1_v, vpaddlq_u16(vpaddlq_u8(v2_2)));
                b1_v = vdotq_u32(b1_v, v2_2, weight_6);
                a1_v = vaddq_u32(a1_v, vpaddlq_u16(vpaddlq_u8(v3_2)));
                b1_v = vdotq_u32(b1_v, v3_2, weight_7);
            }

            let a_combined = vaddq_u32(a0_v, a1_v);
            let b_combined = vaddq_u32(b0_v, b1_v);
            let b_v = vaddq_u32(b_combined, vshlq_n_u32(p_v, 7));

            *a = a.unchecked_add(vaddvq_u32(a_combined));
            *b = vaddvq_u32(b_v);

            blocks_remainder
        }
    }

    #[cfg(target_arch = "arm")]
    #[inline(always)]
    unsafe fn reduce_add_blocks<'a>(a: &mut u32, b: &mut u32, chunk: &'a [u8]) -> &'a [u8] {
        unsafe {
            const BLOCK_SIZE_ARM: usize = 32;

            if chunk.len() < BLOCK_SIZE_ARM {
                return chunk;
            }

            let blocks = chunk.chunks_exact(BLOCK_SIZE_ARM);
            let blocks_remainder = blocks.remainder();

            let weight_lo = get_weight_lo_arm();
            let weight_hi = get_weight_hi_arm();

            #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
            let mut p_acc = a.unchecked_mul(blocks.len() as u32);
            let mut a_acc = 0_u32;
            let mut b_acc = *b;

            for block in blocks {
                use core::mem::transmute;

                let ptr = block.as_ptr();

                let mut sum_a = vdupq_n_u32(0);

                let v_lo = vld1q_u8(ptr);
                let v_hi = vld1q_u8(ptr.add(16));

                let v_lo_wide_lo = vmovl_u8(vget_low_u8(v_lo));
                let v_lo_wide_hi = vmovl_u8(vget_high_u8(v_lo));
                let v_hi_wide_lo = vmovl_u8(vget_low_u8(v_hi));
                let v_hi_wide_hi = vmovl_u8(vget_high_u8(v_hi));

                sum_a = vaddw_u16(sum_a, vget_low_u16(v_lo_wide_lo));
                sum_a = vaddw_u16(sum_a, vget_high_u16(v_lo_wide_lo));
                sum_a = vaddw_u16(sum_a, vget_low_u16(v_lo_wide_hi));
                sum_a = vaddw_u16(sum_a, vget_high_u16(v_lo_wide_hi));
                sum_a = vaddw_u16(sum_a, vget_low_u16(v_hi_wide_lo));
                sum_a = vaddw_u16(sum_a, vget_high_u16(v_hi_wide_lo));
                sum_a = vaddw_u16(sum_a, vget_low_u16(v_hi_wide_hi));
                sum_a = vaddw_u16(sum_a, vget_high_u16(v_hi_wide_hi));

                let sum_array: [u32; 4] = transmute(sum_a);
                let block_sum = sum_array[0]
                    .unchecked_add(sum_array[1])
                    .unchecked_add(sum_array[2])
                    .unchecked_add(sum_array[3]);

                p_acc = p_acc.unchecked_add(a_acc);
                a_acc = a_acc.unchecked_add(block_sum);

                let weight_lo_wide_lo = vmovl_u8(vget_low_u8(weight_lo));
                let weight_lo_wide_hi = vmovl_u8(vget_high_u8(weight_lo));
                let weight_hi_wide_lo = vmovl_u8(vget_low_u8(weight_hi));
                let weight_hi_wide_hi = vmovl_u8(vget_high_u8(weight_hi));

                let mut b_contrib = vdupq_n_u32(0);
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_low_u16(v_lo_wide_lo),
                    vget_low_u16(weight_lo_wide_lo),
                );
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_high_u16(v_lo_wide_lo),
                    vget_high_u16(weight_lo_wide_lo),
                );
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_low_u16(v_lo_wide_hi),
                    vget_low_u16(weight_lo_wide_hi),
                );
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_high_u16(v_lo_wide_hi),
                    vget_high_u16(weight_lo_wide_hi),
                );
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_low_u16(v_hi_wide_lo),
                    vget_low_u16(weight_hi_wide_lo),
                );
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_high_u16(v_hi_wide_lo),
                    vget_high_u16(weight_hi_wide_lo),
                );
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_low_u16(v_hi_wide_hi),
                    vget_low_u16(weight_hi_wide_hi),
                );
                b_contrib = vmlal_u16(
                    b_contrib,
                    vget_high_u16(v_hi_wide_hi),
                    vget_high_u16(weight_hi_wide_hi),
                );

                let b_array: [u32; 4] = transmute(b_contrib);
                b_acc = b_acc
                    .unchecked_add(b_array[0])
                    .unchecked_add(b_array[1])
                    .unchecked_add(b_array[2])
                    .unchecked_add(b_array[3]);
            }

            b_acc = b_acc.unchecked_add(p_acc << 5);

            *a = a.unchecked_add(a_acc);
            *b = b_acc;

            blocks_remainder
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_0() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                64_u8, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_1() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                48_u8, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_2() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                32_u8, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_3() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;
            transmute([16_u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_0_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                128_u8, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_1_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                112_u8, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_2_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                96_u8, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_3_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                80_u8, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_4_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                64_u8, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_5_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                48_u8, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_6_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                32_u8, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    const unsafe fn get_weight_7_128() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;
            transmute([16_u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        }
    }

    #[cfg(target_arch = "arm")]
    #[inline(always)]
    const unsafe fn get_weight_lo_arm() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([
                32_u8, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
            ])
        }
    }

    #[cfg(target_arch = "arm")]
    #[inline(always)]
    const unsafe fn get_weight_hi_arm() -> uint8x16_t {
        unsafe {
            use core::mem::transmute;

            transmute([16_u8, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        }
    }
}

use super::Adler32Imp;

/// Resolves update implementation for ARM NEON
#[must_use]
pub fn get_imp() -> Option<Adler32Imp> { get_imp_inner() }

#[inline]
#[cfg(all(feature = "std", feature = "nightly", target_arch = "arm", target_feature = "v7"))]
fn get_imp_inner() -> Option<Adler32Imp> {
    std::arch::is_arm_feature_detected!("neon").then_some(imp::update as Adler32Imp)
}

#[inline]
#[cfg(all(feature = "std", feature = "nightly", target_arch = "aarch64"))]
fn get_imp_inner() -> Option<Adler32Imp> {
    (std::arch::is_aarch64_feature_detected!("neon")
        && std::arch::is_aarch64_feature_detected!("dotprod"))
    .then_some(imp::update as Adler32Imp)
}

#[inline]
#[cfg(all(
    feature = "nightly",
    target_feature = "neon",
    target_feature = "dotprod",
    not(feature = "std")
))]
fn get_imp_inner() -> Option<Adler32Imp> { Some(imp::update) }

#[inline]
#[cfg(not(any(
    all(feature = "std", feature = "nightly", target_arch = "arm", target_feature = "v7"),
    all(feature = "std", feature = "nightly", target_arch = "aarch64"),
    all(
        feature = "nightly",
        target_feature = "neon",
        target_feature = "dotprod",
        not(feature = "std")
    )
)))]
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
