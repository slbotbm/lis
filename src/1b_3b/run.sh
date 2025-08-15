#!/bin/bash

#gemma3-1B
uv add transformers
uv run gemma3.py

#qwen3-1.7B
uv add transformers==4.52.0
uv run qwen3.py