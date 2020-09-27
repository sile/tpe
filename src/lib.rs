//! This crate provides a hyperparameter optimization algorithm using TPE (Tree-structured Parzen Estimator).
//!
//! # Examples
//!
//! An example optimizing a simple quadratic function which has one numerical and one categorical parameters.
//! ```
//! use rand::SeedableRng as _;
//!
//! # fn main() -> anyhow::Result<()> {
//! let choices = [1, 10, 100];
//! let mut optim0 =
//!     tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(-5.0, 5.0)?);
//! let mut optim1 =
//!     tpe::TpeOptimizer::new(tpe::histogram_estimator(), tpe::categorical_range(choices.len())?);
//!
//! fn objective(x: f64, y: i32) -> f64 {
//!     x.powi(2) + y as f64
//! }
//!
//! let mut best_value = std::f64::INFINITY;
//! let mut rng = rand::rngs::StdRng::from_seed(Default::default());
//! for _ in 0..100 {
//!    let x = optim0.ask(&mut rng)?;
//!    let y = optim1.ask(&mut rng)?;
//!
//!    let v = objective(x, choices[y as usize]);
//!    optim0.tell(x, v)?;
//!    optim1.tell(y, v)?;
//!    best_value = best_value.min(v);
//! }
//!
//! assert_eq!(best_value, 1.000098470725203);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! Please refer to the following papers about the details of TPE:
//!
//! - [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
//! - [Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures](http://proceedings.mlr.press/v28/bergstra13.pdf)
#![warn(missing_docs)]
use crate::density_estimation::{BuildDensityEstimator, DefaultEstimatorBuilder, DensityEstimator};
use crate::range::{Range, RangeError};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::Rng;
use std::num::NonZeroUsize;

pub mod density_estimation;
pub mod range;

/// Creates a `Range` instance.
pub fn range(start: f64, end: f64) -> Result<Range, RangeError> {
    Range::new(start, end)
}

/// Creates a `Range` for a categorical parameter.
///
/// This is equivalent to `range(0.0, cardinality as f64)`.
pub fn categorical_range(cardinality: usize) -> Result<Range, RangeError> {
    Range::new(0.0, cardinality as f64)
}

/// Creates a `DefaultEstimatorBuilder` to build `ParzenEstimator` (for categorical parameter).
pub fn parzen_estimator() -> DefaultEstimatorBuilder {
    DefaultEstimatorBuilder::Parzen(Default::default())
}

/// Creates a `DefaultEstimatorBuilder` to build `HistogramEstimator` (for numerical parameter).
pub fn histogram_estimator() -> DefaultEstimatorBuilder {
    DefaultEstimatorBuilder::Histogram(Default::default())
}

/// Builder of `TpeOptimizer`.
#[derive(Debug)]
pub struct TpeOptimizerBuilder {
    gamma: f64,
    candidates: usize,
}

impl TpeOptimizerBuilder {
    /// Makes a new `TpeOptimizerBuilder` instance with the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the percentage at which the good and bad observations are split.
    ///
    /// The default values is `0.1`.
    pub fn gamma(&mut self, gamma: f64) -> &mut Self {
        self.gamma = gamma;
        self
    }

    /// Sets the number of candidates to be sampled to decide the next parameter.
    ///
    /// The default value is `24`.
    pub fn candidates(&mut self, candidates: usize) -> &mut Self {
        self.candidates = candidates;
        self
    }

    /// Builds a `TpeOptimizer` with the given settings.
    pub fn build<T>(
        &self,
        estimator_builder: T,
        param_range: Range,
    ) -> Result<TpeOptimizer<T>, BuildError>
    where
        T: BuildDensityEstimator,
    {
        if !(0.0 <= self.gamma && self.gamma <= 1.0) {
            return Err(BuildError::GammaOutOfRange);
        }

        Ok(TpeOptimizer {
            param_range,
            estimator_builder,
            trials: Vec::new(),
            is_sorted: false,
            gamma: self.gamma,
            candidates: NonZeroUsize::new(self.candidates).ok_or(BuildError::ZeroCandidates)?,
        })
    }
}

impl Default for TpeOptimizerBuilder {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            candidates: 24,
        }
    }
}

/// Optimizer using TPE.
///
/// This try to search out the parameter value which could minimize the evaluation result.
///
/// Note that an instance of TpeOptimizer can handle only one hyperparameter.
/// So if you want to optimize multiple hyperparameters simultaneously,
/// please create an optimizer for each hyperparameter.
#[derive(Debug)]
pub struct TpeOptimizer<T = DefaultEstimatorBuilder> {
    param_range: Range,
    estimator_builder: T,
    trials: Vec<Trial>,
    is_sorted: bool,
    gamma: f64,
    candidates: NonZeroUsize,
}

impl<T: BuildDensityEstimator> TpeOptimizer<T> {
    /// Makes a new `TpeOptimizer` with the default settings.
    ///
    /// If you want to customize the settings, please use `TpeOptimizerBuilder` instead.
    pub fn new(estimator_builder: T, param_range: Range) -> Self {
        TpeOptimizerBuilder::new()
            .build(estimator_builder, param_range)
            .expect("unreachable")
    }

    /// Returns the next value of the optimization target parameter to be evaluated.
    ///
    /// Note that, before the first asking, it might be worth to give some evaluation
    /// results of randomly sampled observations to `TpeOptimizer` (via the `tell` method)
    /// to reduce bias due to too few samples.
    pub fn ask<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<f64, T::Error> {
        if !self.is_sorted {
            self.trials.sort_by_key(|t| OrderedFloat(t.value));
            self.is_sorted = true;
        }

        let split_point = self.decide_split_point();
        let (superiors, inferiors) = self.trials.split_at(split_point);

        let superior_estimator = self.estimator_builder.build_density_estimator(
            superiors.iter().map(|t| t.param).filter(|p| p.is_finite()),
            self.param_range,
        )?;
        let inferior_estimator = self.estimator_builder.build_density_estimator(
            inferiors.iter().map(|t| t.param).filter(|p| p.is_finite()),
            self.param_range,
        )?;

        let param = (&superior_estimator)
            .sample_iter(rng)
            .take(self.candidates.get())
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

    /// Tells the evaluation result of a hyperparameter value to the optimizer.
    ///
    /// Note that the `param` should be NaN if the hyperparameter was not used in the evaluation
    /// (this could be happen when the entire search space is conditional).
    pub fn tell(&mut self, param: f64, value: f64) -> Result<(), TellError> {
        if value.is_nan() {
            return Err(TellError::NanValue);
        }

        if !(param.is_nan() || self.param_range.contains(param)) {
            return Err(TellError::ParamOutOfRange {
                param,
                range: self.param_range,
            });
        }

        self.trials.push(Trial { param, value });
        self.is_sorted = false;

        Ok(())
    }

    fn decide_split_point(&self) -> usize {
        (self.trials.len() as f64 * self.gamma).ceil() as usize
    }
}

#[derive(Debug, Clone)]
struct Trial {
    param: f64,
    value: f64,
}

/// Possible errors during `TpeOptimizerBuilder::build`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BuildError {
    #[error("the value of `gamma` must be in the range from 0.0 to 1.0")]
    /// The value of `gamma` must be in the range from `0.0` to `1.0`.
    GammaOutOfRange,

    #[error("the value of `candidates` must be a positive integer")]
    /// The value of `candidates` must be a positive integer.
    ZeroCandidates,
}

/// Possible errors during `TpeOptimizer::tell`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TellError {
    #[error("the parameter value {param} is out of the range {range}")]
    /// The parameter value is out of the range.
    ParamOutOfRange {
        /// Actual parameter value.
        param: f64,
        /// Expected parameter range.
        range: Range,
    },

    #[error("NaN value is not allowed")]
    /// NaN value is not allowed.
    NanValue,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn optimizer_works() -> anyhow::Result<()> {
        let choices = [1, 10, 100];
        let mut optim0 = TpeOptimizer::new(parzen_estimator(), range(-5.0, 5.0)?);
        let mut optim1 =
            TpeOptimizer::new(histogram_estimator(), categorical_range(choices.len())?);

        fn objective(x: f64, y: i32) -> f64 {
            x.powi(2) + y as f64
        }

        let mut best_value = std::f64::INFINITY;
        let mut rng = rand::rngs::StdRng::from_seed(Default::default());
        for _ in 0..100 {
            let x = optim0.ask(&mut rng)?;
            let y = optim1.ask(&mut rng)?;

            let v = objective(x, choices[y as usize]);
            optim0.tell(x, v)?;
            optim1.tell(y, v)?;
            best_value = best_value.min(v);
        }
        assert_eq!(best_value, 1.000098470725203);

        Ok(())
    }
}
