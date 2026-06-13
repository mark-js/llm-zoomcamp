FROM debian:trixie-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN groupadd -g 1000 user && \
    useradd -m -s /bin/bash -u 1000 -g 1000 user

USER user

COPY .python-version .
RUN uv python install

WORKDIR /workspaces