//! Optimization benchmark for regression detection and quality estimation.
//!
//! Runs TPE against well-known continuous test functions in two modes:
//!
//! - **Fixed seeds**: deterministic across machines (fixed seeds + `rand` uses
//!   an architecture-independent PRNG). Any change in output signals the
//!   algorithm's behavior changed — useful for regression detection.
//! - **Random seeds**: non-deterministic, fresh OS entropy per seed. Gives an
//!   unbiased estimate of the true optimization quality.
//!
//! The elapsed time is noisy on shared CI runners — it's recorded only as a
//! rough reference, not for strict regression checks.
//!
//! Run with: `cargo run --release --example benchmark`
use rand::Rng;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use std::f64::consts::{E, PI};
use std::time::{Duration, Instant};
use tpe::{TpeOptimizer, parzen_estimator, range};

type Objective = fn(&[f64]) -> f64;

fn sphere(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x * x).sum()
}

fn rosenbrock(xs: &[f64]) -> f64 {
    xs.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn ackley(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let s1 = xs.iter().map(|x| x * x).sum::<f64>();
    let s2 = xs.iter().map(|x| (2.0 * PI * x).cos()).sum::<f64>();
    -20.0 * (-0.2 * (s1 / n).sqrt()).exp() - (s2 / n).exp() + 20.0 + E
}

fn rastrigin(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    10.0 * n
        + xs.iter()
            .map(|x| x * x - 10.0 * (2.0 * PI * x).cos())
            .sum::<f64>()
}

struct Problem {
    name: &'static str,
    dim: usize,
    low: f64,
    high: f64,
    objective: Objective,
}

struct Report {
    best_mean: f64,
    best_std: f64,
    elapsed_secs: f64,
}

fn run_one_seed<R: Rng>(problem: &Problem, trials: usize, rng: &mut R) -> f64 {
    let mut optims: Vec<TpeOptimizer> = (0..problem.dim)
        .map(|_| {
            TpeOptimizer::new(
                parzen_estimator(),
                range(problem.low, problem.high).expect("valid range"),
            )
        })
        .collect();

    let mut xs = vec![0.0f64; problem.dim];
    let mut best = f64::INFINITY;

    for _ in 0..trials {
        for (optim, x) in optims.iter_mut().zip(xs.iter_mut()) {
            *x = optim.ask(rng).expect("ask");
        }
        let v = (problem.objective)(&xs);
        for (optim, &x) in optims.iter_mut().zip(xs.iter()) {
            optim.tell(x, v).expect("tell");
        }
        if v < best {
            best = v;
        }
    }
    best
}

fn summarize(best_values: &[f64], elapsed: Duration) -> Report {
    let n = best_values.len() as f64;
    let mean = best_values.iter().sum::<f64>() / n;
    let variance = best_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    Report {
        best_mean: mean,
        best_std: variance.sqrt(),
        elapsed_secs: elapsed.as_secs_f64(),
    }
}

fn run_fixed(problem: &Problem, trials: usize, seeds: usize) -> Report {
    let start = Instant::now();
    let best_values: Vec<f64> = (0..seeds)
        .map(|seed_n| {
            let mut seed = [0u8; 32];
            seed[0] = seed_n as u8;
            seed[1] = (seed_n >> 8) as u8;
            let mut rng = StdRng::from_seed(seed);
            run_one_seed(problem, trials, &mut rng)
        })
        .collect();
    summarize(&best_values, start.elapsed())
}

fn run_random(problem: &Problem, trials: usize, seeds: usize) -> Report {
    let start = Instant::now();
    let best_values: Vec<f64> = (0..seeds)
        .map(|_| {
            let mut rng = rand::rng();
            run_one_seed(problem, trials, &mut rng)
        })
        .collect();
    summarize(&best_values, start.elapsed())
}

fn print_header() {
    println!(
        "{:<12} {:>3} {:>14} {:>14} {:>9}",
        "problem", "dim", "best_mean", "best_std", "elapsed"
    );
    println!("{}", "-".repeat(56));
}

fn print_row(problem: &Problem, r: &Report) {
    println!(
        "{:<12} {:>3} {:>14.4e} {:>14.4e} {:>8.3}s",
        problem.name, problem.dim, r.best_mean, r.best_std, r.elapsed_secs
    );
}

fn main() {
    let trials = 100;
    let fixed_seeds = 10;
    let random_seeds = 30;

    let problems = [
        Problem {
            name: "sphere",
            dim: 2,
            low: -5.0,
            high: 5.0,
            objective: sphere,
        },
        Problem {
            name: "sphere",
            dim: 5,
            low: -5.0,
            high: 5.0,
            objective: sphere,
        },
        Problem {
            name: "rosenbrock",
            dim: 2,
            low: -5.0,
            high: 10.0,
            objective: rosenbrock,
        },
        Problem {
            name: "rosenbrock",
            dim: 5,
            low: -5.0,
            high: 10.0,
            objective: rosenbrock,
        },
        Problem {
            name: "ackley",
            dim: 2,
            low: -32.768,
            high: 32.768,
            objective: ackley,
        },
        Problem {
            name: "ackley",
            dim: 5,
            low: -32.768,
            high: 32.768,
            objective: ackley,
        },
        Problem {
            name: "rastrigin",
            dim: 2,
            low: -5.12,
            high: 5.12,
            objective: rastrigin,
        },
        Problem {
            name: "rastrigin",
            dim: 5,
            low: -5.12,
            high: 5.12,
            objective: rastrigin,
        },
    ];

    println!("TPE optimization benchmark");
    println!();

    println!("## Fixed seeds (deterministic; for regression detection)");
    println!("trials per seed: {trials}, seeds per problem: {fixed_seeds}");
    println!();
    print_header();
    let mut fixed_total = 0.0;
    for problem in &problems {
        let r = run_fixed(problem, trials, fixed_seeds);
        print_row(problem, &r);
        fixed_total += r.elapsed_secs;
    }
    println!("{}", "-".repeat(56));
    println!("total elapsed: {fixed_total:.3}s");
    println!();

    println!("## Random seeds (non-deterministic; for optimization quality estimate)");
    println!("trials per seed: {trials}, seeds per problem: {random_seeds}");
    println!();
    print_header();
    let mut random_total = 0.0;
    for problem in &problems {
        let r = run_random(problem, trials, random_seeds);
        print_row(problem, &r);
        random_total += r.elapsed_secs;
    }
    println!("{}", "-".repeat(56));
    println!("total elapsed: {random_total:.3}s");
}
