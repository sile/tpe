use anyhow::anyhow;
use kurobako_core::epi::channel::{MessageReceiver, MessageSender};
use kurobako_core::epi::solver::SolverMessage;
use kurobako_core::problem::ProblemSpec;
use kurobako_core::solver::{Capability, SolverSpecBuilder};
use kurobako_core::trial::{IdGen, NextTrial, Params, TrialId};
use rand;
use std::collections::HashMap;

#[derive(Debug)]
struct Solver {
    optimizers: Vec<tpe::TpeOptimizer>,
    evaluating: HashMap<TrialId, Vec<f64>>,
    last_step: u64,
}

impl Solver {
    fn new(problem: &ProblemSpec) -> Self {
        let optimizers = problem
            .params_domain
            .variables()
            .iter()
            .map(|p| {
                tpe::TpeOptimizer::new(
                    tpe::ParzenEstimatorBuilder::default(),
                    tpe::Range::new(p.range().low(), p.range().high()).expect("unreachable"),
                )
            })
            .collect();
        Self {
            optimizers,
            evaluating: HashMap::new(),
            last_step: problem.steps.last(),
        }
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
                random_seed: _,
                problem,
            } => {
                let opt = Solver::new(&problem);
                solvers.insert(solver_id, opt);
            }
            SolverMessage::AskCall {
                solver_id,
                next_trial_id,
            } => {
                let solver = solvers
                    .get_mut(&solver_id)
                    .ok_or_else(|| anyhow!("unknown solver {:?}", solver_id))?;
                // TODO: warmup
                // TODO: Take the random seed into account.
                let mut rng = rand::thread_rng();
                let params = solver
                    .optimizers
                    .iter_mut()
                    .map(|o| o.ask(&mut rng))
                    .collect::<Vec<_>>();
                let mut idg = IdGen::from_next_id(next_trial_id);
                let trial = NextTrial {
                    id: idg.generate(),
                    params: Params::new(params.clone()),
                    next_step: Some(solver.last_step),
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
