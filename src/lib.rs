use crate::density_estimation::{BuildDensityEstimator, DefaultEstimatorBuilder, DensityEstimator};
use crate::range::{Range, RangeError};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::Rng;
use std::num::NonZeroUsize;

pub mod density_estimation;
pub mod range;

pub fn range(start: f64, end: f64) -> Result<Range, RangeError> {
    Range::new(start, end)
}

pub fn parzen_estimator() -> self::density_estimation::ParzenEstimatorBuilder {
    Default::default()
}

pub fn histogram_estimator() -> self::density_estimation::HistogramEstimatorBuilder {
    Default::default()
}

#[derive(Debug)]
pub struct TpeOptions {
    gamma: f64,
    candidates: NonZeroUsize,
}

impl Default for TpeOptions {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            candidates: NonZeroUsize::new(24).expect("unreachable"),
        }
    }
}

#[derive(Debug)]
pub struct TpeOptimizer<T = DefaultEstimatorBuilder> {
    range: Range, // TODO: param_range
    trials: Vec<Trial>,
    is_sorted: bool,
    builder: T,
    options: TpeOptions,
}

impl<T: BuildDensityEstimator> TpeOptimizer<T> {
    pub fn new(builder: T, range: Range) -> Self {
        Self {
            range,
            trials: Vec::new(),
            is_sorted: false,
            builder,
            options: TpeOptions::default(),
        }
    }

    pub fn ask<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<f64, T::Error> {
        if !self.is_sorted {
            self.trials.sort_by_key(|t| OrderedFloat(t.value));
            self.is_sorted = true;
        }

        let split_point = self.decide_split_point();
        let (superiors, inferiors) = self.trials.split_at(split_point);

        let superior_estimator = self.builder.build_density_estimator(
            superiors.iter().map(|t| t.param).filter(|p| p.is_finite()),
            self.range,
        )?;
        let inferior_estimator = self.builder.build_density_estimator(
            inferiors.iter().map(|t| t.param).filter(|p| p.is_finite()),
            self.range,
        )?;

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
        Ok(param)
    }

    fn decide_split_point(&self) -> usize {
        (self.trials.len() as f64 * self.options.gamma).ceil() as usize
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
struct Trial {
    param: f64,
    value: f64,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum TellError {
    #[error("the parameter value {param} is out of the range {range}")]
    ParamOutOfRange { param: f64, range: Range },

    #[error("NaN value is not allowed")]
    NanValue,
}
