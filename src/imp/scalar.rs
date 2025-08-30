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
        assert_eq!(adler32(&[0]), 1 | 1 << 16);
        assert_eq!(adler32(&[0, 0]), 1 | 2 << 16);
        assert_eq!(adler32(&[0; 100]), 0x00640001);
        assert_eq!(adler32(&[0; 1024]), 0x04000001);
        assert_eq!(adler32(&[0; 1024 * 1024]), 0x00F00001);
    }

    #[test]
    fn ones() {
        assert_eq!(adler32(&[0xFF; 1024]), 0x79A6FC2E);
        assert_eq!(
            adler32(&[0xFF; 1024 * 1024]),
            0x8E88EF11
        );
    }

    #[test]
    fn mixed() {
        assert_eq!(adler32(&[1]), 2 | 2 << 16);
        assert_eq!(adler32(&[40]), 41 | 41 << 16);

        assert_eq!(
            adler32(&[0xA5; 1024 * 1024]),
            0xD5009AB1
        );
    }

    /// Example calculation from https://en.wikipedia.org/wiki/Adler-32.
    #[test]
    fn wiki() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    fn adler32(data: &[u8]) -> u32 {
        let (a, b) = super::update(1, 0, data);

        u32::from(b) << 16 | u32::from(a)
    }
}
