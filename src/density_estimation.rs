use crate::Range;
use rand::distributions::Distribution;

pub use self::parzen::{ParzenEstimator, ParzenEstimatorBuilder};

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
