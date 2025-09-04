#[expect(clippy::decimal_literal_representation, reason = "Readability.")]
const MOD: u32 = 65521;
const NMAX: usize = 5552;

#[must_use]
pub fn update(a: u16, b: u16, data: &[u8]) -> (u16, u16) {
    let mut a_ = u32::from(a);

    let mut b_ = u32::from(b);

    let chunks = data.chunks_exact(NMAX);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for byte in chunk {
            a_ = a_.wrapping_add((*byte).into());
            b_ = b_.wrapping_add(a_);
        }

        a_ %= MOD;
        b_ %= MOD;
    }

    for byte in remainder {
        a_ = a_.wrapping_add((*byte).into());
        b_ = b_.wrapping_add(a_);
    }

    a_ %= MOD;
    b_ %= MOD;

    #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
    (a_ as u16, b_ as u16)
}

#[cfg(test)]
mod tests {
    #[test]
    fn zeroes() {
        assert_eq!(adler32(&[]), 1);
        assert_eq!(adler32(&[0]), 1 | (1 << 16_i32));
        assert_eq!(adler32(&[0, 0]), 1 | (2 << 16_i32));
        assert_eq!(adler32(&[0; 100]), 0x0064_0001);
        assert_eq!(adler32(&[0; 1024]), 0x0400_0001);
        assert_eq!(adler32(&[0; 1024 * 1024]), 0x00F0_0001);
    }

    #[test]
    fn ones() {
        assert_eq!(adler32(&[0xFF; 1024]), 0x79A6_FC2E);
        assert_eq!(
            adler32(&[0xFF; 1024 * 1024]),
            0x8E88_EF11
        );
    }

    #[test]
    fn mixed() {
        assert_eq!(adler32(&[1]), 2 | (2 << 16_i32));
        assert_eq!(adler32(&[40]), 41 | (41 << 16_i32));

        assert_eq!(
            adler32(&[0xA5; 1024 * 1024]),
            0xD500_9AB1
        );
    }

    /// Example calculation from <https://en.wikipedia.org/wiki/Adler-32>.
    #[test]
    fn wiki() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    fn adler32(data: &[u8]) -> u32 {
        let (a, b) = super::update(1, 0, data);

        (u32::from(b) << 16_i32) | u32::from(a)
    }
}
