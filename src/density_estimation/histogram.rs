use crate::density_estimation::{BuildDensityEstimator, DensityEstimator};
use crate::Range;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

#[derive(Debug)]
pub struct HistogramEstimatorBuilder {}

impl BuildDensityEstimator for HistogramEstimatorBuilder {
    type Estimator = HistogramEstimator;
    type Error = std::convert::Infallible;

    fn build_density_estimator<I>(
        &self,
        params: I,
        param_range: Range,
    ) -> Result<Self::Estimator, Self::Error>
    where
        I: Iterator<Item = f64> + Clone,
    {
        let cardinality = param_range.width().ceil() as usize;
        let n = params.clone().count() + cardinality;

        let weight = 1.0 / n as f64;
        let mut probabilities = vec![weight; cardinality];
        for p in params {
            probabilities[p.floor() as usize] += weight;
        }

        let distribution = WeightedIndex::new(probabilities.iter()).expect("unreachable");
        Ok(HistogramEstimator {
            probabilities,
            distribution,
        })
    }
}

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
