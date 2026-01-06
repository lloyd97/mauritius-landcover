# Hugging Face Spaces entry point
# This file redirects to our Flask app for Spaces deployment

import os
os.environ['HF_SPACES'] = '1'

from src.web.live_interactive_map import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
