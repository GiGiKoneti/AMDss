import logging
import os

logger = logging.getLogger(__name__)

class VoiceASR:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        if not use_mock:
            # from faster_whisper import WhisperModel
            # self.model = WhisperModel("base", device="cpu", compute_type="int8")
            pass
            
    def transcribe(self, audio_filepath: str) -> str:
        """
        Transcribes the given audio file into text.
        """
        logger.info(f"Transcribing audio file: {audio_filepath}")
        
        if self.use_mock or not os.path.exists(audio_filepath):
            logger.warning("Using mock ASR transcription.")
            return "How does Schrödinger's wave equation relate to quantum entanglement?"
            
        raise NotImplementedError("Real ASR requires faster_whisper installation.")
