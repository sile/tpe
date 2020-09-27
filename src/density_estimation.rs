//! Probability density function estimation.
use crate::Range;
use rand::distributions::Distribution;
use rand::Rng;

pub use self::histogram::{HistogramEstimator, HistogramEstimatorBuilder};
pub use self::parzen::{ParzenEstimator, ParzenEstimatorBuilder};

mod histogram;
mod parzen;

/// This trait allows estimating the probability density of a sample and sampling from the function.
pub trait DensityEstimator: Distribution<f64> {
    /// Estimates the log probability density of a sample.
    fn log_pdf(&self, x: f64) -> f64;
}

/// This trait allows building probability density estimators.
pub trait BuildDensityEstimator {
    /// Density estimator to be built.
    type Estimator: DensityEstimator;

    /// Possible error during building.
    type Error: std::error::Error;

    /// Builds a probability density estimator from the given samples.
    fn build_density_estimator<I>(
        &self,
        xs: I,
        range: Range,
    ) -> Result<Self::Estimator, Self::Error>
    where
        I: Iterator<Item = f64> + Clone;
}

/// Default estimator.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum DefaultEstimator {
    Parzen(ParzenEstimator),
    Histogram(HistogramEstimator),
}

impl DensityEstimator for DefaultEstimator {
    fn log_pdf(&self, x: f64) -> f64 {
        match self {
            Self::Parzen(t) => t.log_pdf(x),
            Self::Histogram(t) => t.log_pdf(x),
        }
    }
}

impl Distribution<f64> for DefaultEstimator {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            Self::Parzen(t) => t.sample(rng),
            Self::Histogram(t) => t.sample(rng),
        }
    }
}

/// Builder of `DefaultEstimator`.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum DefaultEstimatorBuilder {
    Parzen(ParzenEstimatorBuilder),
    Histogram(HistogramEstimatorBuilder),
}

impl BuildDensityEstimator for DefaultEstimatorBuilder {
    type Estimator = DefaultEstimator;
    type Error = std::convert::Infallible;

    fn build_density_estimator<I>(
        &self,
        params: I,
        range: Range,
    ) -> Result<Self::Estimator, Self::Error>
    where
        I: Iterator<Item = f64> + Clone,
    {
        match self {
            Self::Parzen(t) => t
                .build_density_estimator(params, range)
                .map(DefaultEstimator::Parzen),
            Self::Histogram(t) => t
                .build_density_estimator(params, range)
                .map(DefaultEstimator::Histogram),
        }
    }
}
