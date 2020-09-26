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
    pub fn new(start: f64, end: f64) -> Result<Self, RangeError> {
        if !(end - start).is_finite() {
            return Err(RangeError::NonFiniteRange);
        }

        if !(start < end) {
            return Err(RangeError::EmptyRange);
        }

        Ok(Self { start, end })
    }

    pub fn start(self) -> f64 {
        self.start
    }

    pub fn end(self) -> f64 {
        self.end
    }

    pub fn width(self) -> f64 {
        self.end - self.start
    }

    pub fn contains(self, v: f64) -> bool {
        self.start <= v && v < self.end
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum RangeError {
    #[error("not a finite range")]
    NonFiniteRange,

    #[error("an empty range")]
    EmptyRange,
}
