use crate::Range;
use crate::density_estimation::{BuildDensityEstimator, DensityEstimator};
use rand::distr::{Distribution, OpenClosed01};
use rand::seq::IndexedRandom;
use rand::{Rng, RngExt};

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
            let stddev = x.stddev.max(min_stddev).min(max_stddev);
            x.set_stddev(stddev);
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
            .map(Normal::new)
            .collect::<Vec<_>>();
        xs.sort_by(|x, y| x.mean.total_cmp(&y.mean));

        self.setup_stddev(&mut xs, range);

        let p_accept = xs
            .iter()
            .map(|x| x.cdf(range.end()) - x.cdf(range.start()))
            .sum::<f64>()
            / xs.len() as f64;

        // log(1 / (n * p_accept)), added to each log_pdf result.
        let log_offset = -(xs.len() as f64).ln() - p_accept.ln();

        Ok(ParzenEstimator {
            samples: xs,
            range,
            log_offset,
        })
    }
}

/// Normal distribution.
#[derive(Debug)]
struct Normal {
    mean: f64,
    stddev: f64,
    stddev_inv: f64,
    log_norm_const: f64,
}

impl Normal {
    fn new(mean: f64) -> Self {
        Self {
            mean,
            stddev: f64::NAN,
            stddev_inv: f64::NAN,
            log_norm_const: f64::NAN,
        }
    }

    fn set_stddev(&mut self, stddev: f64) {
        self.stddev = stddev;
        self.stddev_inv = 1.0 / stddev;
        self.log_norm_const = -0.5 * (2.0 * std::f64::consts::PI).ln() - stddev.ln();
    }

    #[inline]
    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) * self.stddev_inv;
        -0.5 * z * z + self.log_norm_const
    }

    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf((x - self.mean) * self.stddev_inv / std::f64::consts::SQRT_2))
    }
}

// Error function approximation by Abramowitz & Stegun 7.1.26 (max error ~1.5e-7).
fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Parzen window based density estimator.
///
/// This can be used for numerical parameters.
#[derive(Debug)]
pub struct ParzenEstimator {
    samples: Vec<Normal>,
    range: Range,
    log_offset: f64,
}

impl DensityEstimator for ParzenEstimator {
    fn log_pdf(&self, x: f64) -> f64 {
        // Streaming `logsumexp` over `Normal::log_pdf` values, skipping the
        // intermediate `Vec` allocation the non-streaming form would require.
        let mut samples = self.samples.iter();
        let first = samples
            .next()
            .expect("ParzenEstimator always has at least one sample");
        let mut max_lp = first.log_pdf(x);
        let mut sum_exp = 1.0;
        for sample in samples {
            let lp = sample.log_pdf(x);
            if lp > max_lp {
                sum_exp = sum_exp * (max_lp - lp).exp() + 1.0;
                max_lp = lp;
            } else {
                sum_exp += (lp - max_lp).exp();
            }
        }
        max_lp + sum_exp.ln() + self.log_offset
    }
}

impl Distribution<f64> for ParzenEstimator {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        while let Some(x) = self.samples.choose(rng) {
            // Box-Muller transform: convert two uniform samples into a standard normal sample.
            // `OpenClosed01` excludes 0 so that `ln(u1)` is finite.
            let u1: f64 = rng.sample(OpenClosed01);
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let draw = x.mean + x.stddev * z;
            if self.range.contains(draw) {
                return draw;
            }
        }
        unreachable!();
    }
}
