# Renders docs/report/report.html to docs/report.pdf with WeasyPrint.
# Pango and the fonts live in this image, so `make report` behaves the same on
# Windows, macOS and Linux hosts.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
        fonts-noto-core fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir weasyprint==70.0

# Fontconfig needs a writable cache directory when run as a non-root user.
ENV HOME=/tmp
WORKDIR /docs
ENTRYPOINT ["weasyprint"]
