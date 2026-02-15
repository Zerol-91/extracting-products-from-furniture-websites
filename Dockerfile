FROM python:3.10-slim
# To see training logs immediately, and not when the buffer is full
ENV PYTHONUNBUFFERED=1 
WORKDIR /app

# 2. Installing system utilities
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 3. uv installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Copying dependency files
COPY pyproject.toml uv.lock ./

# 5. Installing dependencies 
RUN uv sync --no-install-project

# 6. Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# 7. Copy ALL code (src and scripts) and data (data)
# data required for training
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/ 

# 8. Training during container assembly
# Result (папка models/final_model) remains into the image.
RUN python -m scripts.research.train

# 9. Final Launch Command (API)
# The API will load the model already located in models/final_model
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]