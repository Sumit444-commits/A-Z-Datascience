#!/bin/bash

# Set your Conda environment name
ENV_NAME="stream_no_T"  # Replace with your actual Conda environment name

echo "🚀 Activating Conda environment: $ENV_NAME..."
conda activate "$ENV_NAME"

echo "✅ Installing FFmpeg and ImageMagick..."
conda install -c conda-forge ffmpeg imagemagick -y

echo "🔍 Checking FFmpeg installation..."
ffmpeg -version || { echo "❌ FFmpeg installation failed!"; exit 1; }

echo "🔍 Checking ImageMagick installation..."
magick -version || { echo "❌ ImageMagick installation failed!"; exit 1; }

# Find the ImageMagick binary path
MAGICK_PATH=$(where magick | head -n 1)

if [ -z "$MAGICK_PATH" ]; then
    echo "❌ ImageMagick binary not found!"
    exit 1
else
    echo "✅ Found ImageMagick binary at: $MAGICK_PATH"
fi

# Update MoviePy's configuration
echo "🔧 Setting ImageMagick path for MoviePy..."
python -c "from moviepy.config import change_settings; change_settings({'IMAGEMAGICK_BINARY': r'$MAGICK_PATH'})"

echo "✅ Fix applied successfully! Restart your terminal and test MoviePy."
