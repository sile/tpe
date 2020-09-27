use crate::density_estimation::{BuildDensityEstimator, DensityEstimator};
use crate::Range;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

/// Builder of `HistogramEstimator`.
#[derive(Debug, Default)]
pub struct HistogramEstimatorBuilder {}

impl HistogramEstimatorBuilder {
    /// Makes a new `HistogramEstimatorBuilder` instance.
    pub fn new() -> Self {
        Self::default()
    }
}

impl BuildDensityEstimator for HistogramEstimatorBuilder {
    type Estimator = HistogramEstimator;
    type Error = std::convert::Infallible;

    fn build_density_estimator<I>(
        &self,
        xs: I,
        range: Range,
    ) -> Result<Self::Estimator, Self::Error>
    where
        I: Iterator<Item = f64> + Clone,
    {
        let cardinality = range.width().ceil() as usize;
        let n = xs.clone().count() + cardinality;

        let weight = 1.0 / n as f64;
        let mut probabilities = vec![weight; cardinality];
        for x in xs {
            probabilities[x.floor() as usize] += weight;
        }

        let distribution = WeightedIndex::new(probabilities.iter()).expect("unreachable");
        Ok(HistogramEstimator {
            probabilities,
            distribution,
        })
    }
}

/// Histogram based density estimation.
///
/// This can be used for categorical parameters.
///
/// Note that this estimator assumes that each told value is the index, not the raw value,
/// of a categorical parameter.
#[derive(Debug)]
pub struct HistogramEstimator {
    probabilities: Vec<f64>,
    distribution: WeightedIndex<f64>,
}

impl DensityEstimator for HistogramEstimator {
    fn log_pdf(&self, x: f64) -> f64 {
        self.probabilities[x.floor() as usize]
    }
}

impl Distribution<f64> for HistogramEstimator {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.distribution.sample(rng) as f64
    }
}
