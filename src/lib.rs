//! # simd-adler32
//!
//! A SIMD-accelerated Adler-32 hash algorithm implementation.
//!
//! ## Features
//!
//! - No dependencies
//! - Support `no_std` (with `default-features = false`)
//! - Runtime CPU feature detection (when `std` enabled)
//! - Blazing fast performance on as many targets as possible
//! - Default to scalar implementation when simd not available
//!
//! ## Quick start
//!
//! > Cargo.toml
//!
//! ```toml
//! [dependencies]
//! simd-adler32 = "*"
//! ```
//!
//! > example.rs
//!
//! ```rust
//! use simd_adler32::Adler32;
//!
//! let mut adler = Adler32::new();
//! adler.write(b"rust is pretty cool, man");
//! let hash = adler.finish();
//!
//! println!("{}", hash);
//! // 1921255656
//! ```
//!
//! ## Feature flags
//!
//! * `std` - Enabled by default
//!
//! Enables std support, see [CPU Feature Detection](#cpu-feature-detection) for
//! runtime detection support.
//! * `nightly`
//!
//! Enables nightly features required for avx512 support.
//!
//! * `const-generics` - Enabled by default
//!
//! Enables const-generics support allowing for user-defined array hashing by
//! value.  See [`Adler32Hash`] for details.
//!
//! ## Support
//!
//! **CPU Features**
//!
//! | impl | arch             | feature |
//! | ---- | ---------------- | ------- |
//! | ✅   | `x86`, `x86_64`  | avx512  |
//! | ✅   | `x86`, `x86_64`  | avx2    |
//! | ✅   | `x86`, `x86_64`  | ssse3   |
//! | ✅   | `x86`, `x86_64`  | sse2    |
//! | ✅   | `aarch64`        | neon    |
//! | 🚧   | `arm`            | neon    |
//! | ✅   | `wasm32`         | simd128 |
//!
//! **MSRV** `1.36.0`\*\*
//!
//! Minimum supported rust version is tested before a new version is published.
//! [**] Feature `const-generics` needs to disabled to build on rustc versions
//! `<1.51` which can be done by updating your dependency definition to the
//! following.
//!
//! ## CPU Feature Detection
//! simd-adler32 supports both runtime and compile time CPU feature detection
//! using the `std::arch::is_x86_feature_detected` macro when the `Adler32`
//! struct is instantiated with the `new` fn.
//!
//! Without `std` feature enabled simd-adler32 falls back to compile time
//! feature detection using `target-feature` or `target-cpu` flags supplied to
//! rustc. See: [https://rust-lang.github.io/packed_simd/perf-guide/target-feature/rustflags.html](
//! https://rust-lang.github.io/packed_simd/perf-guide/target-feature/rustflags.html)
//! for more information.

#![warn(
    clippy::all,
    clippy::cargo,
    clippy::complexity,
    clippy::correctness,
    clippy::nursery,
    clippy::pedantic,
    clippy::perf,
    clippy::restriction,
    clippy::style,
    clippy::suspicious
)]
#![allow(clippy::as_conversions, reason = "Redundant with specific checks.")]
#![allow(
    clippy::blanket_clippy_restriction_lints,
    reason = "Overly verbose to individually enable."
)]
#![allow(clippy::similar_names, reason = "Convention.")]
#![allow(clippy::implicit_return, reason = "Follow Rust idiomatic returns.")]
#![allow(clippy::inline_always, reason = "Intended.")]
#![allow(clippy::large_stack_arrays, reason = "Used only in tests.")]
#![allow(clippy::large_stack_frames, reason = "Used only in tests.")]
#![allow(clippy::min_ident_chars, reason = "Convention.")]
#![allow(clippy::missing_inline_in_public_items, reason = "Not beneficial.")]
#![allow(clippy::mod_module_files, reason = "Maintain existing layout.")]
#![allow(clippy::multiple_unsafe_ops_per_block, reason = "Readability.")]
#![allow(clippy::undocumented_unsafe_blocks, reason = "Intrinsics.")]
#![allow(
    clippy::single_call_fn,
    reason = "Single use functions are used for clarity and composability."
)]
#![allow(clippy::separated_literal_suffix, reason = "Desired style.")]
#![allow(clippy::single_char_lifetime_names, reason = "Convention.")]

//! Feature detection tries to use the fastest supported feature first.
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(
    all(feature = "nightly", target_arch = "arm"),
    feature(arm_target_feature, stdarch_arm_feature_detection, stdarch_arm_neon_intrinsics)
)]
#![cfg_attr(all(feature = "nightly", target_arch = "aarch64"), feature(stdarch_neon_dotprod))]

#[doc(hidden)]
pub mod hash;
#[doc(hidden)]
pub mod imp;

#[cfg(feature = "std")]
pub mod bufread {
    //! BufRead-based hashing.
    //!
    //! Separate `BufRead` trait implemented to allow for custom buffer size
    //! optimization.
    //!
    //! # Example
    //! ```rust
    //! use std::io::{BufReader, Cursor};
    //!
    //! use simd_adler32::bufread::adler32;
    //!
    //! let mut reader = Cursor::new(b"Hello there");
    //! let mut reader = BufReader::new(reader);
    //! let hash = adler32(&mut reader).unwrap();
    //!
    //! println!("{}", hash) // 800813569
    //! ```
    use std::io::{BufRead, ErrorKind, Result};

    use crate::Adler32;

    /// Compute Adler-32 hash on buf reader until EOF.
    ///
    /// # Errors
    ///
    /// Returns a non-recoverable IO error; that is, not:
    /// - `ErrorKind::Interrupted`
    /// - `ErrorKind::UnexpectedEof`
    ///
    /// # Example
    /// ```rust
    /// use std::io::{BufReader, Cursor};
    ///
    /// use simd_adler32::bufread::adler32;
    ///
    /// let mut reader = Cursor::new(b"Hello there");
    /// let mut reader = BufReader::new(reader);
    /// let hash = adler32(&mut reader).unwrap();
    ///
    /// println!("{}", hash) // 800813569
    /// ```
    pub fn adler32<R: BufRead>(reader: &mut R) -> Result<u32> {
        let mut hash = Adler32::new();

        loop {
            let consumed = match reader.fill_buf() {
                | Ok(buf) => {
                    if buf.is_empty() {
                        return Ok(hash.finish());
                    }

                    hash.write(buf);
                    buf.len()
                },
                | Err(err) => match err.kind() {
                    | ErrorKind::Interrupted => continue,
                    | ErrorKind::UnexpectedEof => return Ok(hash.finish()),
                    | _ => return Err(err),
                },
            };

            reader.consume(consumed);
        }
    }
}

#[cfg(feature = "std")]
pub mod read {
    //! Reader-based hashing.
    //!
    //! # Example
    //! ```rust
    //! use std::io::Cursor;
    //!
    //! use simd_adler32::read::adler32;
    //!
    //! let mut reader = Cursor::new(b"Hello there");
    //! let hash = adler32(&mut reader).unwrap();
    //!
    //! println!("{}", hash) // 800813569
    //! ```
    use std::io::{Read, Result};

    use crate::Adler32;

    /// Compute Adler-32 hash on reader until EOF.
    ///
    /// # Errors
    ///
    /// Returns an IO error which may be recoverable.
    ///
    /// # Example
    /// ```rust
    /// use std::io::Cursor;
    ///
    /// use simd_adler32::read::adler32;
    ///
    /// let mut reader = Cursor::new(b"Hello there");
    /// let hash = adler32(&mut reader).unwrap();
    ///
    /// println!("{}", hash) // 800813569
    /// ```
    pub fn adler32<R: Read>(reader: &mut R) -> Result<u32> {
        let mut hash = Adler32::new();
        let mut buf = [0; 4096];

        loop {
            match reader.read(&mut buf) {
                | Ok(0) => return Ok(hash.finish()),
                | Ok(n) => {
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "A panic would mean the Read trait has been implemented \
                                  incorrectly."
                    )]
                    hash.write(&buf[..n]);
                },
                | Err(err) => return Err(err),
            }
        }
    }
}

/// An adler32 hash generator type.
#[derive(Clone)]
pub struct Adler32 {
    /// The low-order 16 bits of the sum.
    a: u16,
    /// The high-order 16 bits of the sum.
    b: u16,
}

impl Adler32 {
    /// Return the hash value for the values written so far.
    ///
    /// Despite its name, the method does not reset the hasher’s internal state.
    /// Additional writes will continue from the current value. If you need
    /// to start a fresh hash value, you will have to use `reset`.
    #[must_use]
    pub fn finish(&self) -> u32 { (u32::from(self.b) << 16) | u32::from(self.a) }

    /// Construct a new `Adler32` using existing checksum.
    ///
    /// # Examples
    /// ```rust
    /// use simd_adler32::Adler32;
    ///
    /// let mut adler = Adler32::from_checksum(0xDEAD_BEAF);
    /// ```
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "Intended.")]
    pub const fn from_checksum(checksum: u32) -> Self {
        Self {
            a: checksum as u16,
            b: (checksum >> 16) as u16,
        }
    }

    /// Construct a new `Adler32`.
    ///
    /// Potential overhead here due to runtime feature detection; however,
    /// testing on 100k and 10k random byte arrays shows little-to-no impact.
    ///
    /// # Examples
    /// ```rust
    /// use simd_adler32::Adler32;
    ///
    /// let mut adler = Adler32::new();
    /// ```
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Reset the internal state.
    pub const fn reset(&mut self) {
        self.a = 1;
        self.b = 0;
    }

    /// Computes hash for supplied data and store results in an internal state.
    pub fn write(&mut self, data: &[u8]) {
        let (a, b) = imp::call(self.a, self.b, data);

        self.a = a;
        self.b = b;
    }
}

/// An Adler-32 hash-able type.
pub trait Adler32Hash {
    /// Feeds this value into `Adler32`.
    fn hash(&self) -> u32;
}

impl Default for Adler32 {
    fn default() -> Self { Self { a: 1, b: 0 } }
}

/// Compute Adler-32 hash on `Adler32Hash` type.
///
/// # Arguments
/// * `hash` - A Adler-32 hash-able type.
///
/// # Examples
/// ```rust
/// use simd_adler32::adler32;
///
/// let hash = adler32(b"Adler-32");
/// println!("{}", hash); // 800813569
/// ```
pub fn adler32<H: Adler32Hash>(hash: &H) -> u32 { hash.hash() }

#[cfg(test)]
mod tests {
    #[test]
    fn from_checksum_works() {
        let buf = b"rust is pretty cool man";
        let sum = 0xDEAD_BEAF;

        let mut simd = super::Adler32::from_checksum(sum);
        let mut adler = adler::Adler32::from_checksum(sum);

        simd.write(buf);
        adler.write_slice(buf);

        let simd_f = simd.finish();
        let scalar = adler.checksum();

        assert_eq!(simd_f, scalar);
    }
}
