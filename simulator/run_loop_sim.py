from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_loop
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.torax_pydantic import model_config
from torax._src.config import config_loader
import xarray as xr


def make_step_fn(
    torax_config: model_config.ToraxConfig,
) -> step_function.SimulationStepFn:
  """Prepare a TORAX step function from a config."""
  geometry_provider = torax_config.geometry.build_provider
  physics_models = torax_config.build_physics_models()

  solver = torax_config.solver.build_solver(
      physics_models=physics_models,
  )

  runtime_params_provider = (
      build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
  )

  return step_function.SimulationStepFn(
      solver=solver,
      time_step_calculator=torax_config.time_step_calculator.time_step_calculator,
      geometry_provider=geometry_provider,
      runtime_params_provider=runtime_params_provider,
  )


def prepare_simulation(
    torax_config: model_config.ToraxConfig,
) -> tuple[
    sim_state.SimState,
    post_processing.PostProcessedOutputs,
    step_function.SimulationStepFn,
]:
  """Prepare a TORAX simulation returning the necessary inputs for the run loop.

  Args:
    torax_config: The TORAX config to use for the simulation.

  Returns:
    A tuple containing:
      - The initial state.
      - The initial post processed outputs.
      - The simulation step function.
  """
  step_fn = make_step_fn(torax_config)

  if torax_config.restart and torax_config.restart.do_restart:
    initial_state, post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=torax_config.restart,
            step_fn=step_fn,
        )
    )
  else:
    initial_state, post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs(
            step_fn=step_fn,
        )
    )

  return (
      initial_state,
      post_processed_outputs,
      step_fn,
  )


def run_simulation(
    torax_config: model_config.ToraxConfig,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
) -> tuple[xr.DataTree, output.StateHistory]:
  """Runs a TORAX simulation using the config and returns the outputs.

  Args:
    torax_config: The TORAX config to use for the simulation.
    log_timestep_info: Whether to log the timestep information.
    progress_bar: Whether to show a progress bar.

  Returns:
    A tuple of the simulation outputs in the form of a DataTree and the state
    history which is intended for helpful use with debugging as it contains
    the `CoreProfiles`, `CoreTransport`, `CoreSources`, `Geometry`, and
    `PostProcessedOutputs` dataclasses for each step of the simulation.
  """

  (
      initial_state,
      post_processed_outputs,
      step_fn,
  ) = prepare_simulation(torax_config)


  state_history, post_processed_outputs_history, sim_error = run_loop.run_loop(
      initial_state=initial_state,
      initial_post_processed_outputs=post_processed_outputs,
      step_fn=step_fn,
      log_timestep_info=log_timestep_info,
      progress_bar=progress_bar,
  )

  state_history = output.StateHistory(
      state_history=state_history,
      post_processed_outputs_history=post_processed_outputs_history,
      sim_error=sim_error,
      torax_config=torax_config,
  )

  return (
      state_history.simulation_output_to_xr(),
      state_history,
  )
# config = config_loader.build_torax_config_from_file('//iterhybrid_rampup.py')
# print(config)
#
# run_simulation(
#     torax_config=config
# )