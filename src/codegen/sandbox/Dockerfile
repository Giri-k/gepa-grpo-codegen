# Sandbox image: plain Python plus the worker script. No project code, data or
# expected outputs go in here. Build context is this directory only.
# Pinned by digest (multi-arch index: works on Mac arm64, Vista arm64, Lonestar6 amd64).
FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin runner
COPY worker.py /sandbox/worker.py

USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-s", "/sandbox/worker.py"]
