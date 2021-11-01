FROM continuumio/miniconda3 AS build

COPY enviornment.yml .
RUN conda env create -f enviornment.yml

#install conda-pack for creating archives of conda-env
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
RUN conda-pack -n realtime -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack


# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM debian:buster AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# When image is run, run the code with the environment
# activated:
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && \
           python new_main.py