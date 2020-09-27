use crate::Range;
use rand::distributions::Distribution;
use rand::Rng;

pub use self::histogram::{HistogramEstimator, HistogramEstimatorBuilder};
pub use self::parzen::{ParzenEstimator, ParzenEstimatorBuilder};

mod histogram;
mod parzen;

pub trait DensityEstimator: Distribution<f64> {
    fn log_pdf(&self, x: f64) -> f64;
}

pub trait BuildDensityEstimator {
    type Estimator: DensityEstimator;
    type Error: std::error::Error;

    fn build_density_estimator<I>(
        &self,
        params: I,
        range: Range,
    ) -> Result<Self::Estimator, Self::Error>
    where
        I: Iterator<Item = f64> + Clone;
}

#[derive(Debug)]
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

#[derive(Debug)]
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
