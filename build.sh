#!/bin/bash
# Install ffmpeg for audio conversion (used directly in main.py)
apt-get update && apt-get install -y ffmpeg
pip install -r requirements.txt

