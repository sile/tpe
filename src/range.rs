//! Parameter range.

/// Range with a start (inclusive) and end (exclusive) as floating-point values.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    start: f64,
    end: f64,
}

impl std::fmt::Display for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl Range {
    /// Makes a new [`Range`] instance.
    pub fn new(start: f64, end: f64) -> Result<Self, RangeError> {
        if !(end - start).is_finite() {
            return Err(RangeError::NonFiniteRange);
        }

        if start.partial_cmp(&end) != Some(std::cmp::Ordering::Less) {
            return Err(RangeError::EmptyRange);
        }

        Ok(Self { start, end })
    }

    /// Returns the start point of the range.
    pub fn start(self) -> f64 {
        self.start
    }

    /// Returns the end point of the range.
    pub fn end(self) -> f64 {
        self.end
    }

    /// Width of the range.
    pub fn width(self) -> f64 {
        self.end - self.start
    }

    /// Return `true` if the given point is contained in the range, otherwise `false`.
    pub fn contains(self, v: f64) -> bool {
        self.start <= v && v < self.end
    }
}

/// Possible errors during [`Range`] construction.
#[derive(Debug, Clone)]
pub enum RangeError {
    /// Not a finite range.
    NonFiniteRange,

    /// An empty range.
    EmptyRange,
}

impl std::fmt::Display for RangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RangeError::NonFiniteRange => write!(f, "not a finite range"),
            RangeError::EmptyRange => write!(f, "an empty range"),
        }
    }
}

impl std::error::Error for RangeError {}
