use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::distribution::{Continuous, Univariate};
use std::num::NonZeroUsize;

#[derive(Debug, Default)]
pub struct ParzenEstimatorBuilder {}

impl ParzenEstimatorBuilder {
    fn setup_stddev(&self, xs: &mut [Entry], range: ParamRange) {
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

    fn build<I>(&self, xs: I, range: ParamRange) -> Self::Estimator
    where
        I: Iterator<Item = f64> + Clone,
    {
        let prior = (range.start() + range.end()) * 0.5;
        let mut xs = xs
            .chain(std::iter::once(prior))
            .map(|x| Entry {
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

        ParzenEstimator {
            samples: xs,
            range,
            p_accept,
        }
    }
}

// TODO: Normal
#[derive(Debug)]
struct Entry {
    mean: f64,
    stddev: f64,
}

impl Entry {
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

#[derive(Debug)]
pub struct ParzenEstimator {
    samples: Vec<Entry>,
    range: ParamRange,
    p_accept: f64,
}

impl EstimateDensity for ParzenEstimator {
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

    pub fn width(self) -> f64 {
        self.end - self.start
    }

    pub fn contains(self, v: f64) -> bool {
        self.start <= v && v < self.end
    }
}

#[derive(Debug)]
pub struct TpeOptimizer<T = ParzenEstimatorBuilder> {
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

        let superior_estimator = self.builder.build(
            superiors.iter().map(|t| t.param).filter(|p| p.is_finite()),
            self.range,
        );
        let inferior_estimator = self.builder.build(
            inferiors.iter().map(|t| t.param).filter(|p| p.is_finite()),
            self.range,
        );

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
