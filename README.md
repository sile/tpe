tpe
===

[![tpe](https://img.shields.io/crates/v/tpe.svg)](https://crates.io/crates/tpe)
[![Documentation](https://docs.rs/tpe/badge.svg)](https://docs.rs/tpe)
[![Actions Status](https://github.com/sile/tpe/workflows/CI/badge.svg)](https://github.com/sile/tpe/actions)
[![Coverage Status](https://coveralls.io/repos/github/sile/tpe/badge.svg?branch=master)](https://coveralls.io/github/sile/tpe?branch=master)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This crate provides a hyperparameter optimization algorithm using TPE (Tree-structured Parzen Estimator).


Examples
--------

### Minimize the result of a quadratic function

An example optimizing a simple quadratic function which has one numerical and one categorical parameters.
```rust
use rand::SeedableRng as _;

let choices = [1, 10, 100];
let mut optim0 =
    tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(-5.0, 5.0)?);
let mut optim1 =
    tpe::TpeOptimizer::new(tpe::histogram_estimator(), tpe::categorical_range(choices.len())?);

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
```

### [`kurobako`] benchmark

There is an example [examples/tpe-solver.rs](examples/tpe-solver.rs) which implements
the [`kurobako`] solver interface, so you can run a benchmark using TPE as follows:

```console
$ PROBLEMS=$(kurobako problem-suite sigopt auc)
$ SOLVERS="$(kurobako solver command -- cargo run --release --example tpe-solver) $(kurobako solver optuna)"
$ kurobako studies --solvers $SOLVERS --problems $PROBLEMS --repeats 30 --budget 80 | kurobako run > result.json
$ cat result.json | kurobako report > report.md
```

[`kurobako`]: https://github.com/sile


References
----------

Please refer to the following papers about the details of TPE:

- [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
- [Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures](http://proceedings.mlr.press/v28/bergstra13.pdf)
