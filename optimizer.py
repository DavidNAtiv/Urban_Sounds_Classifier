import logging
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformIntegerParameterRange)

# trying to load Bayesian optimizer package
try:
    from clearml.automation.optuna import OptimizerOptuna  # noqa
    aSearchStrategy = OptimizerOptuna
except ImportError as ex:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB  # noqa
        aSearchStrategy = OptimizerBOHB
    except ImportError as ex:
        logging.getLogger().warning(
            'Apologies, it seems you do not have \'optuna\' or \'hpbandster\' installed, '
            'we will be using RandomSearch strategy instead')
        aSearchStrategy = RandomSearch


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='Hyper-Parameter Optimization',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

# experiment template to optimize in the hyper-parameter optimization
args = {
    'template_task_id': None,
    'run_as_service': False,
}
args = task.connect(args)

# Get the template task experiment that we want to optimize
if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(
        project_name = "Urban Sound Classifier", task_name = "04 Testing HP Optim").id

# Set default queue name for the Training tasks themselves.
# later can be overridden in the UI
execution_queue = 'first_queue'

# Example use case:
an_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=args['template_task_id'],
    # here we define the hyper-parameters to optimize
    # Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
    # For Example, here we see in the base experiment a section Named: "General"
    # under it a parameter named "batch_size", this becomes "General/batch_size"
    # If you have `argparse` for example, then arguments will appear under the "Args" section,
    # and you should instead pass "Args/batch_size"
    hyper_parameters=[
        #UniformIntegerParameterRange('Args/dropout', min_value=0, max_value=0.3, step_size=0.3),
        DiscreteParameterRange('Args/lr', values=[1e-4, 1e-3]),
        DiscreteParameterRange('Args/batch', values=[32, 64]),

        DiscreteParameterRange('Args/epochs', values=[4]),
        DiscreteParameterRange('Args/episods', values=[4]),
        DiscreteParameterRange('Args/eval', values=[4]),
        DiscreteParameterRange('Args/log', values=[1]),
    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title='Loss',
    objective_metric_series='loss',
    # now we decide if we want to maximize it or minimize it (accuracy we maximize)
    objective_metric_sign='min',
    # let us limit the number of concurrent experiments,
    # this in turn will make sure we do dont bombard the scheduler with experiments.
    # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
    max_number_of_concurrent_tasks=1,
    # this is the optimizer class (actually doing the optimization)
    # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
    # more are coming soon...
    optimizer_class=aSearchStrategy,
    # Select an execution queue to schedule the experiments for execution
    execution_queue=execution_queue,
    # If specified all Tasks created by the HPO process will be created under the `spawned_project` project
    spawn_project=None,  # 'HPO spawn project',
    # If specified only the top K performing Tasks will be kept, the others will be automatically archived
    save_top_k_tasks_only=None,  # 5,
    # Optional: Limit the execution time of a single experiment, in minutes.
    # (this is optional, and if using  OptimizerBOHB, it is ignored)
    time_limit_per_job=10.,
    # Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
    # assuming a single experiment is usually hours...
    pool_period_min=0.2,
    # set the maximum number of jobs to launch for the optimization, default (None) unlimited
    # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
    # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
    total_max_jobs=4,
    # set the minimum number of iterations for an experiment, before early stopping.
    # Does not apply for simple strategies such as RandomSearch or GridSearch
    min_iteration_per_job=1,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=30,
)

# if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
if args['run_as_service']:
    # if this code is executed by `clearml-agent` the function call does nothing.
    # if executed locally, the local process will be terminated, and a remote copy will be executed instead
    task.execute_remotely(queue_name='services', exit_process=True)

# report every 12 seconds, this is way too often, but we are testing here J
an_optimizer.set_report_period(2.2)
# start the optimization process, callback function to be called every time an experiment is completed
# this function returns immediately
an_optimizer.start(job_complete_callback=job_complete_callback)
# You can also use the line below instead to run all the optimizer tasks locally, without using queues or agent
# an_optimizer.start_locally(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)
an_optimizer.set_time_limit(in_minutes=20.0)
# wait until process is done (notice we are controlling the optimization process in the background)
an_optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
# make sure background optimization stopped
an_optimizer.stop()

print('We are done, good bye')
