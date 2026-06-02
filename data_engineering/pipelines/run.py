from datetime import datetime as dt
from pathlib import Path

import click

from data_engineering.pipelines.etl import etl_pipeline


@click.command(
    help="""
Main entry point for the pipeline execution. 
This entrypoint is where everything comes together.

Run the pipelines with various options.

Examples:

  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache
  
  \b
  # Run only the ETL pipeline
  python run.py --only-etl

"""
)
@click.option(
    "--run-etl",
    is_flag=True,
    default=False,
    help="Whether to run the ETL pipeline.",
)
def main(
    run_etl: bool = False,
    etl_config_filename: str = "digital_data_etl_lilian_weng.yaml",
) -> None:
    assert run_etl, "Please specify action to run"

    pipeline_args = {}

    root_dir = Path(__file__).resolve().parent.parent

    if run_etl:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "config" / etl_config_filename
        assert pipeline_args["config_path"].exists(), (
            f"Config file not found: {pipeline_args['config_path']}"
        )
        pipeline_args["run_name"] = (
            f"digital_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        etl_pipeline.with_options(**pipeline_args)(**run_args_etl)

if __name__ == "__main__":
    main()