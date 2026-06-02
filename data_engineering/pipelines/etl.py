"""
Provide a name and this will run the whole ETL pipeline, which will extract data from the source, transform it, and load it.
1) crawl links looking for the name
2)
"""

import functools

import yaml

from data_engineering.crawlers.crawl_links import crawl_links
from data_engineering.pipelines.get_or_create_user import get_or_create_user


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

@pipeline
def etl_pipeline(user_full_name: str, links: list[str]) -> None:
    user = get_or_create_user(user_full_name)
    crawl_links(user=user, links=links)
    return

