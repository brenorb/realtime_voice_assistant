import asyncio
import json
import runpy
import sys
import unittest
from contextlib import asynccontextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "real_time_voice.py"


class Closed(Exception):
    pass


def audio_module(sounddevice=None, websockets=None):
    with patch.dict(sys.modules, {
        "sounddevice": sounddevice or SimpleNamespace(),
        "websockets": websockets or SimpleNamespace(),
        "numpy": SimpleNamespace(),
        "dotenv": SimpleNamespace(load_dotenv=Mock()),
    }):
        return runpy.run_path(str(SCRIPT), run_name="fixture_voice")


class AudioShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_close_releases_both_audio_streams(self):
        input_opened = asyncio.Event()
        output_started = asyncio.Event()
        input_closed = Mock()
        socket_closed = Mock()

        class InputStream:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                input_opened.set()
                return self

            def __exit__(self, *_args):
                input_closed()

        output = SimpleNamespace(start=output_started.set, stop=Mock(), close=Mock())
        devices = SimpleNamespace(InputStream=InputStream, OutputStream=Mock(return_value=output))

        class Socket:
            created = False

            async def recv(self):
                if not self.created:
                    self.created = True
                    return json.dumps({"type": "session.created"})
                await input_opened.wait()
                await output_started.wait()
                raise Closed()

            async def send(self, message):
                assert json.loads(message)["type"] == "session.update"

        @asynccontextmanager
        async def connect(*_args, **_kwargs):
            try:
                yield Socket()
            finally:
                socket_closed()

        module = audio_module(devices, SimpleNamespace(
            connect=connect, exceptions=SimpleNamespace(ConnectionClosed=Closed),
        ))
        streamer = module["AudioStreamer"]("fixture-only", 0, 1)
        with redirect_stdout(StringIO()):
            await asyncio.wait_for(streamer.start(), timeout=1)

        self.assertFalse(streamer.should_record)
        input_closed.assert_called_once()
        output.stop.assert_called_once()
        output.close.assert_called_once()
        socket_closed.assert_called_once()

    async def test_failed_output_stop_still_closes_the_stream(self):
        module = audio_module()
        audio = module["AudioOut"](24000, 1, 0)
        stream = SimpleNamespace(stop=Mock(side_effect=OSError("fixture stop failure")), close=Mock())
        audio.stream = stream
        with self.assertRaises(OSError):
            await audio.stop()
        stream.close.assert_called_once()
        self.assertIsNone(audio.stream)
