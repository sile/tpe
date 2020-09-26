use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::Rng;
use std::num::NonZeroUsize;

pub trait EstimateDensity: Distribution<f64> {
    fn log_pdf(&self, x: f64) -> f64;
}

pub trait BuildDensityEstimator {
    type Estimator: EstimateDensity;

    fn build<I>(&self, params: I, range: ParamRange) -> Self::Estimator
    where
        I: Iterator<Item = f64> + Clone;
}

#[derive(Debug)]
pub struct TpeOptions {
    candidates: NonZeroUsize,
}

impl Default for TpeOptions {
    fn default() -> Self {
        Self {
            candidates: NonZeroUsize::new(24).expect("unreachable"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParamRange {
    start: f64,
    end: f64,
}

impl std::fmt::Display for ParamRange {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl ParamRange {
    pub fn new(start: f64, end: f64) -> Result<Self, ParamRangeError> {
        if !(end - start).is_finite() {
            return Err(ParamRangeError::NonFiniteRange);
        }

        if !(start < end) {
            return Err(ParamRangeError::EmptyRange);
        }

        Ok(Self { start, end })
    }

    pub fn start(self) -> f64 {
        self.start
    }

    pub fn end(self) -> f64 {
        self.end
    }

    pub fn contains(self, v: f64) -> bool {
        self.start <= v && v < self.end
    }
}

#[derive(Debug)]
pub struct TpeOptimizer<T> {
    range: ParamRange,
    trials: Vec<Trial>,
    is_sorted: bool,
    builder: T,
    options: TpeOptions,
}

impl<T: BuildDensityEstimator> TpeOptimizer<T> {
    pub fn new(builder: T, range: ParamRange) -> Self {
        Self {
            range,
            trials: Vec::new(),
            is_sorted: false,
            builder,
            options: TpeOptions::default(),
        }
    }

    pub fn ask<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        if !self.is_sorted {
            self.trials.sort_by_key(|t| OrderedFloat(t.value));
            self.is_sorted = true;
        }

        let split_point = self.decide_split_point();
        let (superiors, inferiors) = self.trials.split_at(split_point);

        let superior_estimator = self
            .builder
            .build(superiors.iter().map(|t| t.param), self.range);
        let inferior_estimator = self
            .builder
            .build(inferiors.iter().map(|t| t.param), self.range);

        let param = (&superior_estimator)
            .sample_iter(rng)
            .take(self.options.candidates.get())
            .map(|candidate| {
                let superior_log_likelihood = superior_estimator.log_pdf(candidate);
                let inferior_log_likelihood = inferior_estimator.log_pdf(candidate);
                let ei = superior_log_likelihood - inferior_log_likelihood;
                (ei, candidate)
            })
            .max_by_key(|(ei, _)| OrderedFloat(*ei))
            .map(|(_, param)| param)
            .expect("unreachable");
        param
    }

    fn decide_split_point(&self) -> usize {
        (self.trials.len() as f64 * 0.1).ceil() as usize
    }

    pub fn tell(&mut self, param: f64, value: f64) -> Result<(), TellError> {
        if value.is_nan() {
            return Err(TellError::NanValue);
        }

        if !(param.is_nan() || self.range.contains(param)) {
            return Err(TellError::ParamOutOfRange {
                param,
                range: self.range,
            });
        }

        self.trials.push(Trial { param, value });
        self.is_sorted = false;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Trial {
    pub param: f64,
    pub value: f64,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum TellError {
    #[error("the parameter value {param} is out of the range {range}")]
    ParamOutOfRange { param: f64, range: ParamRange },

    #[error("NaN value is not allowed")]
    NanValue,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ParamRangeError {
    #[error("not a finite range")]
    NonFiniteRange,

    #[error("an empty range")]
    EmptyRange,
}
