# Package the app on a slim 3.12 base.
#
# Why 3.12 (not 3.13): Gradio needs the stdlib `audioop` module, which was
# removed in Python 3.13. Pinning python:3.12-slim keeps the UI runnable.
FROM python:3.12-slim

WORKDIR /app

# --- Layer caching ---------------------------------------------------------
# Copy ONLY requirements first and install, THEN copy the source. Dependencies
# change rarely; source changes often. Keeping the expensive `pip install` on
# its own layer (keyed to requirements.txt) means a code edit reuses the cached
# dependency layer instead of reinstalling every package -- rebuilds drop from
# minutes to seconds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the source. A change here invalidates only this layer, not the deps above.
COPY . .

# Gradio serves on 7860.
EXPOSE 7860

# Bind to 0.0.0.0 so the server is reachable from the host (inside a container
# the default 127.0.0.1 would only be reachable from within the container).
ENV GRADIO_SERVER_NAME=0.0.0.0

# The API key is provided at RUN time, never baked into the image:
#   docker build -t cs-support .
#   docker run --rm -p 7860:7860 --env-file .env cs-support
#
# Launch the UNIFIED app (verify + interrupt + memory + music/invoice/policy
# routing behind one graph), not the supervisor-only UI (app.py).
CMD ["python", "app_unified.py"]
