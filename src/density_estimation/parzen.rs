use crate::density_estimation::{BuildDensityEstimator, DensityEstimator};
use crate::Range;
use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::distribution::{Continuous, Univariate};

/// Builder of [`ParzenEstimator`].
#[derive(Debug, Default)]
pub struct ParzenEstimatorBuilder {}

impl ParzenEstimatorBuilder {
    /// Makes a new [`ParzenEstimatorBuilder`] instance.
    pub fn new() -> Self {
        Self::default()
    }

    fn setup_stddev(&self, xs: &mut [Normal], range: Range) {
        let n = xs.len();
        for i in 0..n {
            let prev = if i == 0 {
                range.start()
            } else {
                xs[i - 1].mean
            };
            let curr = xs[i].mean;
            let succ = xs.get(i + 1).map_or(range.end(), |x| x.mean);
            xs[i].stddev = (curr - prev).max(succ - curr);
        }

        if n >= 2 {
            xs[0].stddev = xs[1].mean - xs[0].mean;
            xs[n - 1].stddev = xs[n - 1].mean - xs[n - 2].mean;
        }

        let max_stddev = range.width();
        let min_stddev = range.width() / 100f64.min(1.0 + n as f64);
        for x in xs {
            x.stddev = x.stddev.max(min_stddev).min(max_stddev);
        }
    }
}

impl BuildDensityEstimator for ParzenEstimatorBuilder {
    type Estimator = ParzenEstimator;
    type Error = std::convert::Infallible;

    fn build_density_estimator<I>(
        &self,
        xs: I,
        range: Range,
    ) -> Result<Self::Estimator, Self::Error>
    where
        I: Iterator<Item = f64> + Clone,
    {
        let prior = (range.start() + range.end()) * 0.5;
        let mut xs = xs
            .chain(std::iter::once(prior))
            .map(|x| Normal {
                mean: x,
                stddev: std::f64::NAN,
            })
            .collect::<Vec<_>>();
        xs.sort_by_key(|x| OrderedFloat(x.mean));

        self.setup_stddev(&mut xs, range);

        let p_accept = xs
            .iter()
            .map(|x| x.cdf(range.end()) - x.cdf(range.start()))
            .sum::<f64>()
            / xs.len() as f64;

        Ok(ParzenEstimator {
            samples: xs,
            range,
            p_accept,
        })
    }
}

/// Normal distribution.
#[derive(Debug)]
struct Normal {
    mean: f64,
    stddev: f64,
}

impl Normal {
    fn log_pdf(&self, x: f64) -> f64 {
        statrs::distribution::Normal::new(self.mean, self.stddev)
            .expect("unreachable")
            .ln_pdf(x)
    }

    fn cdf(&self, x: f64) -> f64 {
        statrs::distribution::Normal::new(self.mean, self.stddev)
            .expect("unreachable")
            .cdf(x)
    }
}

/// Parzen window based density estimator.
///
/// This can be used for numerical parameters.
#[derive(Debug)]
pub struct ParzenEstimator {
    samples: Vec<Normal>,
    range: Range,
    p_accept: f64,
}

impl DensityEstimator for ParzenEstimator {
    fn log_pdf(&self, x: f64) -> f64 {
        let weight = 1.0 / self.samples.len() as f64;
        let xs = self
            .samples
            .iter()
            .map(|sample| sample.log_pdf(x) + (weight / self.p_accept).ln())
            .collect::<Vec<_>>();
        logsumexp(&xs)
    }
}

fn logsumexp(xs: &[f64]) -> f64 {
    let max_x = xs
        .iter()
        .max_by_key(|&&x| OrderedFloat(x))
        .expect("unreachable");
    xs.iter().map(|&x| (x - max_x).exp()).sum::<f64>().ln() + max_x
}

impl Distribution<f64> for ParzenEstimator {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        while let Some(x) = self.samples.choose(rng) {
            let draw = rand_distr::Normal::new(x.mean, x.stddev)
                .expect("unreachable")
                .sample(rng);
            if self.range.contains(draw) {
                return draw;
            }
        }
        unreachable!();
    }
}
