import functools

import yaml


def pipeline(fn):
    """decorator that adds pipeline configs"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    def with_options(**pipeline_args):
        config_path = pipeline_args.get("config_path")
        params = {}
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                params = config.get("parameters", {})

        def configured_run(**run_args):
            merged = {**params, **run_args}  # run_args override YAML
            print(f"[{pipeline_args.get('run_name')}] Starting pipeline...")
            return fn(**merged)

        return configured_run

    wrapper.with_options = with_options
    return wrapper
