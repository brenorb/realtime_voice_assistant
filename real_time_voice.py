# Import necessary libraries
import asyncio  # For asynchronous programming
import base64  # For encoding/decoding audio data
import json  # For handling JSON data
import os  # For interacting with the operating system
import websockets  # For WebSocket communication
import sounddevice as sd  # For audio input/output
import numpy as np  # For numerical operations
import threading  # For thread-based parallelism
from dotenv import load_dotenv

NAME = "João da Silva"

# Class to handle audio output
class AudioOut:
    def __init__(self, sample_rate, channels, output_device_id):
        # Initialize audio output parameters
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_device_id = output_device_id
        self.audio_buffer = bytearray()  # Buffer to store audio data
        self.audio_buffer_lock = asyncio.Lock()  # Lock for thread-safe access to the buffer
        self.audio_playback_queue = asyncio.Queue()  # Queue for audio playback
        self.stream = None  # Audio output stream

    async def start(self):
        # Start the audio output stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=self._audio_callback,
            device=self.output_device_id,
            latency='low'
        )
        self.stream.start()
        await self._playback_loop()  # Start the playback loop

    def _audio_callback(self, outdata, frames, time, status):
        # Callback function for the audio output stream
        if status:
            print(status)
        bytes_to_read = frames * self.channels * 2
        with threading.Lock():
            # Read data from the buffer and write to the output
            if len(self.audio_buffer) >= bytes_to_read:
                data = self.audio_buffer[:bytes_to_read]
                del self.audio_buffer[:bytes_to_read]
            else:
                # If not enough data, pad with zeros
                data = self.audio_buffer + bytes([0] * (bytes_to_read - len(self.audio_buffer)))
                self.audio_buffer.clear()
        outdata[:] = np.frombuffer(data, dtype='int16').reshape(-1, self.channels)

    async def _playback_loop(self):
        # Continuous loop to process audio chunks from the queue
        while True:
            chunk = await self.audio_playback_queue.get()
            if chunk is None:
                continue
            async with self.audio_buffer_lock:
                self.audio_buffer.extend(chunk)

    async def add_audio(self, chunk):
        # Add audio chunk to the playback queue
        await self.audio_playback_queue.put(chunk)

    async def clear_audio(self):
        # Clear the audio playback queue and buffer
        while not self.audio_playback_queue.empty():
            try:
                self.audio_playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        async with self.audio_buffer_lock:
            self.audio_buffer.clear()

    async def stop(self):
        # Stop and close the audio output stream
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

# Class to handle the main audio streaming functionality
class AudioStreamer:
    def __init__(self, api_key, input_device_id, output_device_id):
        # Initialize the audio streamer with API key and device IDs
        self.api_key = api_key
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        self.sample_rate = 24000
        self.channels = 1
        self.chunk_duration = 0.02  # 20ms chunks
        self.audio_format = 'int16'
        self.should_record = True
        self.audio_out = AudioOut(self.sample_rate, self.channels, self.output_device_id)

    async def start(self):
        # Start the audio streaming session
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        async with websockets.connect(self.url, additional_headers=headers) as ws:
            print("Connected to the OpenAI Realtime API.")

            # Wait for session creation confirmation
            event = await ws.recv()
            event_data = json.loads(event)
            if event_data["type"] == "session.created":
                print("Session initialized.")

            # Configure session parameters
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": {
                        "type": "server_vad"
                    },
                    "instructions": "Você é uma secretária pessoal chamada Bia recebendo uma ligação. Seu chefe está ocupado no momento."
                    f"Não infome o nome do seu chefe em nenhum momento. Pergunte para quem é a ligação e, se não for para `{NAME}`, desligue a ligação."
                    "Não informe fale este nome em nenhum memomento, apenas confirme caso seja quem a outra pessoa procura."
                    "Após, a confirmação, pergunte o assunto e pegue as informações necessárias."
                    "Após anotar, pergunte se há algo a mais, e peça um contato para retorno para seu chefe poder retornar a ligação."
                }
            }))

            self.should_record = True
            tasks = (
                asyncio.create_task(self.receive_events(ws)),
                asyncio.create_task(self.audio_out.start()),
                asyncio.create_task(self.send_audio(ws)),
            )

            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                self.should_record = False
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await self.audio_out.stop()

    async def send_audio(self, ws):
        # Function to send audio data to the API
        print("Start speaking to the assistant (Press Ctrl+C to exit).")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time, status):
            # Callback function for audio input
            if not self.should_record:
                return
            if status:
                print(status, flush=True)
            audio_bytes = indata.tobytes()
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
            message_event = {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio
            }
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(message_event)), loop)

        # Set up and start the audio input stream
        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                            dtype=self.audio_format, callback=callback,
                            blocksize=int(self.sample_rate * self.chunk_duration),
                            device=self.input_device_id):
            while self.should_record:
                await asyncio.sleep(self.chunk_duration)

    async def receive_events(self, ws):
        # Function to receive and process events from the API
        while True:
            try:
                response = await ws.recv()
                event = json.loads(response)

                if event["type"] == "response.audio.delta":
                    # Handle incoming audio data
                    audio_chunk = base64.b64decode(event["delta"])
                    await self.audio_out.add_audio(audio_chunk)
                elif event["type"] == "response.audio.done":
                    # Handle end of audio response
                    await self.audio_out.add_audio(None)
                    print("Response complete.")
                elif event["type"] == "input_audio_buffer.speech_started":
                    # Handle start of user speech
                    await ws.send(json.dumps({
                        "type": "response.cancel"
                    }))
                    await self.audio_out.clear_audio()
                    print("User started speaking. Clearing audio playback.")
                elif event["type"] == "input_audio_buffer.speech_stopped":
                    # Handle end of user speech
                    print("User stopped speaking.")
                elif event["type"] == "error":
                    # Handle error events
                    error = event.get("error", {})
                    message = error.get("message", "")
                    if message != "Error committing input audio buffer: the buffer is empty.":
                        print(f"Error: {message}")
                else:
                    pass
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed.")
                break

def select_audio_device(input_output):
    # Function to let user select audio input/output device
    devices = sd.query_devices()
    print(f"Available {input_output} audio devices:")
    for i, device in enumerate(devices):
        if (input_output == 'input' and device['max_input_channels'] > 0) or \
           (input_output == 'output' and device['max_output_channels'] > 0):
            print(f"{i}: {device['name']}")
    return int(input(f"Enter the number of the {input_output} device you want to use: "))

def main():
    # Load .env file before getting API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")

    input_device_id = select_audio_device('input')
    output_device_id = select_audio_device('output')

    streamer = AudioStreamer(api_key, input_device_id, output_device_id)
    
    try:
        asyncio.run(streamer.start())
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        # Optional: Add any cleanup code here
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
