# Real-Time Voice Assistant

This project implements a real-time voice assistant using OpenAI's API. It allows for bidirectional audio communication with an AI model, enabling voice-based interactions through a web interface.

## Features

- Real-time audio streaming to and from OpenAI's API
- Automatic speech detection
- Audio playback of AI responses
- Web-based user interface
- Conversation history display

## Requirements

- Python 3.7+
- Websockets
- Sounddevice
- NumPy
- Python-dotenv
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   uv init
   uv add -r requirements.txt
   ```

3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the application:

```
uv run real_time_voice.py
```