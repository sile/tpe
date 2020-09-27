//! A TPE based solver implementation for the [`kurobako`] benchmark.
//!
//! [`kurobako`]: https://github.com/sile/kurobako
use anyhow::anyhow;
use kurobako_core::domain::{self, Variable};
use kurobako_core::epi::channel::{MessageReceiver, MessageSender};
use kurobako_core::epi::solver::SolverMessage;
use kurobako_core::problem::ProblemSpec;
use kurobako_core::rng::ArcRng;
use kurobako_core::solver::{Capability, SolverSpecBuilder};
use kurobako_core::trial::{IdGen, NextTrial, Params, TrialId};
use std::collections::HashMap;

#[derive(Debug)]
struct Solver {
    problem: ProblemSpec,
    optimizers: Vec<tpe::TpeOptimizer>,
    evaluating: HashMap<TrialId, Vec<f64>>,
    rng: ArcRng,
}

impl Solver {
    fn new(problem: ProblemSpec, seed: u64) -> anyhow::Result<Self> {
        let optimizers = problem
            .params_domain
            .variables()
            .iter()
            .map(Self::create_optimizer)
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            problem,
            optimizers,
            evaluating: HashMap::new(),
            rng: ArcRng::new(seed),
        })
    }

    fn create_optimizer(param: &Variable) -> anyhow::Result<tpe::TpeOptimizer> {
        match param.range() {
            domain::Range::Continuous { low, high } => match param.distribution() {
                domain::Distribution::Uniform => Ok(tpe::TpeOptimizer::new(
                    tpe::parzen_estimator(),
                    tpe::range(*low, *high)?,
                )),
                domain::Distribution::LogUniform => Ok(tpe::TpeOptimizer::new(
                    tpe::parzen_estimator(),
                    tpe::range(low.ln(), high.ln())?,
                )),
            },
            domain::Range::Discrete { low, high } => Ok(tpe::TpeOptimizer::new(
                tpe::parzen_estimator(),
                tpe::range(*low as f64, *high as f64)?,
            )),
            domain::Range::Categorical { choices } => Ok(tpe::TpeOptimizer::new(
                tpe::histogram_estimator(),
                tpe::categorical_range(choices.len())?,
            )),
        }
    }

    fn unwarp(&self, params: &[f64]) -> Vec<f64> {
        self.problem
            .params_domain
            .variables()
            .iter()
            .zip(params.iter().copied())
            .map(|(p, v)| match p.range() {
                domain::Range::Continuous { .. } => match p.distribution() {
                    domain::Distribution::Uniform => v,
                    domain::Distribution::LogUniform => v.exp(),
                },
                domain::Range::Discrete { .. } => v.floor(),
                domain::Range::Categorical { .. } => v,
            })
            .collect()
    }

    fn warp(&self, params: &[f64]) -> Vec<f64> {
        self.problem
            .params_domain
            .variables()
            .iter()
            .zip(params.iter().copied())
            .map(|(p, v)| match p.range() {
                domain::Range::Continuous { .. } => match p.distribution() {
                    domain::Distribution::Uniform => v,
                    domain::Distribution::LogUniform => v.ln(),
                },
                domain::Range::Discrete { .. } => v + 0.5,
                domain::Range::Categorical { .. } => v,
            })
            .collect()
    }
}

fn main() -> anyhow::Result<()> {
    let stdout = std::io::stdout();
    let stdin = std::io::stdin();
    let mut tx = MessageSender::new(stdout.lock());
    let mut rx = MessageReceiver::<SolverMessage, _>::new(stdin.lock());

    let spec = SolverSpecBuilder::new("TPE")
        .capable(Capability::Categorical)
        .capable(Capability::Concurrent)
        .capable(Capability::LogUniformContinuous)
        .capable(Capability::UniformContinuous)
        .capable(Capability::UniformDiscrete)
        .finish();
    tx.send(&SolverMessage::SolverSpecCast { spec })?;

    let mut solvers = HashMap::new();
    loop {
        match rx.recv()? {
            SolverMessage::CreateSolverCast {
                solver_id,
                random_seed,
                problem,
            } => {
                let opt = Solver::new(problem, random_seed)?;
                solvers.insert(solver_id, opt);
            }
            SolverMessage::AskCall {
                solver_id,
                next_trial_id,
            } => {
                let solver = solvers
                    .get_mut(&solver_id)
                    .ok_or_else(|| anyhow!("unknown solver {:?}", solver_id))?;

                let rng = &mut solver.rng;
                let params = solver
                    .optimizers
                    .iter_mut()
                    .map(|o| o.ask(rng).map_err(anyhow::Error::from))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let params = solver.unwarp(&params);

                let mut idg = IdGen::from_next_id(next_trial_id);
                let trial = NextTrial {
                    id: idg.generate(),
                    params: Params::new(params.clone()),
                    next_step: Some(solver.problem.steps.last()),
                };
                solver.evaluating.insert(trial.id, params);

                tx.send(&SolverMessage::AskReply {
                    next_trial_id: idg.peek_id().get(),
                    trial,
                })?;
            }
            SolverMessage::TellCall { solver_id, trial } => {
                let solver = solvers
                    .get_mut(&solver_id)
                    .ok_or_else(|| anyhow!("unknown solver {:?}", solver_id))?;
                let params = solver.evaluating.remove(&trial.id).expect("unreachable");
                let params = solver.warp(&params);
                for (o, p) in solver.optimizers.iter_mut().zip(params.into_iter()) {
                    o.tell(p, trial.values[0])?;
                }
                tx.send(&SolverMessage::TellReply {})?;
            }
            SolverMessage::DropSolverCast { solver_id } => {
                solvers.remove(&solver_id);
            }
            other => panic!("unexpected message: {:?}", other),
        }
    }
}
