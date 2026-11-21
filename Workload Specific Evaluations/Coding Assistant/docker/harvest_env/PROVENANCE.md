# Provenance

`docker_allowlist.py` and `egress_proxy/` are copied without changes from
[`aws-samples/sample-code-eval-harvester`](https://github.com/aws-samples/sample-code-eval-harvester),
`harvest_env/src/harvest_env/`. `__init__.py` is new: the original package also holds a Lambda
MicroVMs environment that this module does not use. Licence: MIT No Attribution, the same licence as
this repository.

`docker_allowlist.py` subclasses internals of Harbor's Docker environment. It was written against
Harbor 0.22.0 and tested with Harbor 0.24.0. If a later Harbor version changes
`DockerEnvironment._docker_compose_paths`, `_apply_network_policy`, or `_run_docker_compose_command`,
copy the files again from the source repository.

Docker Compose builds the proxy image (`harvest-env-egress-proxy:1`) from `egress_proxy/` the first
time that it is used. There is no prebuilt image to pull.
