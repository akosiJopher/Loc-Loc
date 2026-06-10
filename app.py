# Secrets — NEVER commit real camera/admin credentials
.streamlit/secrets.toml
.env

# Model weights (large; auto-download on first run)
*.pt

# Runtime / generated files
latest_frame.jpg
static/
counts.json
frame_meta.json
*.tmp

# Python
__pycache__/
*.py[cod]
.venv/
venv/
