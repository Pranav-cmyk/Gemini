from dataclasses import dataclass, field
from os import getenv
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import (
    Tool, ToolConfig, FunctionCallingConfig, FunctionCallingConfigMode,
    GoogleSearch, ToolCodeExecution, SpeechConfig, VoiceConfig,
    PrebuiltVoiceConfig, GenerateContentConfig, LiveConnectConfig,
    GenerationConfig, Part
)
from rich.console import Console
from rich.markdown import Markdown
from rich.logging import RichHandler
from sounddevice import OutputStream, InputStream
from numpy import (
    frombuffer, concatenate, float32, ndarray
)
from typing import List, Any, Optional
from torch import device, cuda, tensor, float32
from warnings import filterwarnings
from whisper import load_model
import time
import logging

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("Gemini-Logger")


load_dotenv()

@dataclass
class GeminiContentConfig:
    """Configuration for Gemini content generation.
    
    Attributes:
        temperature (float): Controls randomness in generation (0.0-1.0)
        max_output_tokens (int): Maximum number of tokens in generated output
        top_k (int): Number of highest probability tokens to consider
        top_p (float): Cumulative probability threshold for token selection
        candidate_count (int): Number of alternative generations to produce
        system_instruction (str): System-level instruction for the model
        tools (List[Tool]): List of tools available to the model
        tool_config (ToolConfig): Configuration for tool usage
        response_schema (Optional[Any]): Schema for structuring responses
        response_mime_type (Optional[str]): Expected MIME type of responses
    """
    temperature: float = 0.9
    max_output_tokens: int = 8024
    top_k: int = 10
    top_p: float = 0.9
    candidate_count: int = 1
    system_instruction: str = "You are a helpful friendly assistant."
    tools: List[Any] = field(default_factory=lambda: [
        Tool(google_search=GoogleSearch()),
        Tool(code_execution=ToolCodeExecution())
    ])
    tool_config: ToolConfig = field(default_factory=lambda: ToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode=FunctionCallingConfigMode.ANY
        )
    ))
    response_schema: Optional[Any] = None
    response_mime_type: Optional[str] = None

    def to_generate_content_config(self) -> GenerateContentConfig:
        """Convert to GenerateContentConfig object."""
        return GenerateContentConfig(**self.__dict__)

@dataclass
class GeminiLiveConfig:
    """Configuration for Gemini live interactions.
    
    Attributes:
        response_modalities (List[str]): Supported response types
        speech_config (SpeechConfig): Configuration for speech generation
        generation_config (GenerationConfig): Parameters for content generation
        tools (List[Tool]): Available tools for the model
    """
    response_modalities: List[str] = field(default_factory=lambda: ["Audio"])
    speech_config: SpeechConfig = field(default_factory=lambda: SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Puck")
        )
    ))
    generation_config: GenerationConfig = field(default_factory=lambda: GenerationConfig(
        temperature=0.9,
        max_output_tokens=1024,
        top_k=10,
        top_p=0.9,
    ))
    tools: List[Tool] = field(default_factory=lambda: [Tool(google_search=GoogleSearch())])

    def to_live_connect_config(self) -> LiveConnectConfig:
        """Convert to LiveConnectConfig object."""
        return LiveConnectConfig(**self.__dict__)

class GeminiError(Exception):
    """Base exception class for Gemini-related errors."""
    pass

class GeminiConfigError(GeminiError):
    """Raised when there's an error in Gemini configuration."""
    pass

class FileProcessingError(GeminiError):
    """Raised when there's an error processing files."""
    pass

class VideoProcessingError(GeminiError):
    """Raised when there's an error processing video."""
    pass

class AudioProcessingError(GeminiError):
    """Raised when there's an error processing audio."""
    pass

class TranscriptionError(GeminiError):
    """Raised when there's an error transcribing audio."""
    pass

class GoogleGemini:
    """
    A class to interact with Google's Gemini AI model.
    
    Attributes:
        model (str): The Gemini model identifier
        console (Console): Rich console for formatted output
        content_config (GenerateContentConfig): Configuration for content generation
        live_config (LiveConnectConfig): Configuration for live interactions
        chat: Active chat session
        client (Client): Google AI client instance
    """

    ALLOWED_VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: str = getenv("GOOGLE_API_KEY"),
        content_config: Optional[GeminiContentConfig] = None,
        live_config: Optional[GeminiLiveConfig] = None
    ):
        """
        Initialize the GoogleGemini instance.
        
        Args:
            model: Gemini model identifier
            api_key: Google API key
            content_config: Configuration for content generation
            live_config: Configuration for live interactions
            
        Raises:
            GeminiConfigError: If client initialization fails
        """
        if not api_key:
            raise GeminiConfigError("API key not found")
            
        self.model = model
        self.console = Console()
        self.content_config = (content_config or GeminiContentConfig()).to_generate_content_config()
        self.live_config = (live_config or GeminiLiveConfig()).to_live_connect_config()
        self.chat = None
        self._initialize_client(api_key)
        self.device_ = device("cuda" if cuda.is_available() else "cpu")
        filterwarnings("ignore", category=FutureWarning)
        self._initalize_whisper_model()
        
    def _initalize_whisper_model(self) -> None:
        try:
            self.whisper_model = load_model('turbo').to(self.device_)
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise GeminiConfigError(f"Error initializing Whisper model: {e}")
        
    def _initialize_client(self, api_key: str) -> None:
        """Initialize the Google AI client."""
        try:
            self.client = Client(
                api_key=api_key,
                http_options={"api_version": "v1alpha"},
            )
            logger.info("Client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            raise GeminiConfigError(f"Error initializing client: {e}")

    def conversational_assistant(
        self,
        files: Optional[List[str]] = None,
        youtube_url: Optional[str] = None,
        initial_prompt: str = "You are a helpful assistant following the system's instructions.",
    ) -> None:
        """
        Start a conversational assistant session.
        
        Args:
            files: List of file paths to process
            youtube_url: YouTube video URL to process
            initial_prompt: Initial system prompt
            
        Raises:
            FileProcessingError: If file processing fails
            VideoProcessingError: If video processing fails
        """
        try:
            uploaded_files = None
            video = None
            
            if files:
                uploaded_files = self._process_files(files)
            if youtube_url:
                video = self._process_url(youtube_url)
                
            self._initialize_conversation(initial_prompt, files=uploaded_files, url=video)
            self._setup_chat_loop()
            
            if uploaded_files:
                self._delete_files(uploaded_files)
                
        except Exception as e:
            logger.error(f"Conversation assistant error: {e}")
            raise

    def _process_files(self, files: List[str]) -> List[Any]:
        """Process and upload files."""
        uploaded_files = []
        try:
            for file_path in files:
                uploaded_file = self.client.files.upload(file=file_path)
                
                if file_path.lower().endswith(('.mp4', '.mp3', '.wav')):
                    self._wait_for_processing(uploaded_file)
                
                uploaded_files.append(uploaded_file)
                logger.info(f"Successfully processed file: {file_path}")
                
            return uploaded_files
        except Exception as e:
            logger.error(f"File processing error: {e}")
            raise FileProcessingError(f"Error uploading files: {e}")

    def _wait_for_processing(self, file: Any, timeout: int = 300) -> None:
        """Wait for file processing to complete."""
        start_time = time.time()
        while file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                raise FileProcessingError("File processing timeout")
            time.sleep(2)
            file = self.client.files.get(name=file.name)
            
        if file.state.name == "FAILED":
            raise FileProcessingError(f"File processing failed: {file.error}")

    async def start_voice_interaction(self, 
        voice: str = "Puck",
        voice_input_enabled: bool = False,
        recording_duration: int = 5,
        system_instructions: str = "You are a helpful assistant."
    ) -> None:
        """
        Start voice-based interaction with Gemini.
        
        Args:
            voice: Name of the voice to use, allowed_voices are ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
            prompt: Initial prompt
            voice_input_enabled: Whether to enable voice input
            recording_duration: Duration for voice recording in seconds
            
        Raises:
            ValueError: If invalid voice selected
            AudioProcessingError: If audio processing fails
        """
        if voice not in self.ALLOWED_VOICES:
            raise ValueError(f"Invalid voice selection. Allowed voices: {self.ALLOWED_VOICES}")
            
        self.live_config.speech_config.voice_config.prebuilt_voice_config.voice_name = voice
        
        try:
            async with self.client.aio.live.connect(
                model=self.model,
                config=self.live_config
            ) as session:
                
                logger.info("Configuring voice interaction...")
                await session.send(input=system_instructions, end_of_turn=False)
                logger.info("Voice interaction configured successfully")
                
                await self._handle_voice_interaction(
                    session, voice_input_enabled, recording_duration
                )
                
        except Exception as e:
            logger.error(f"Voice interaction error: {e}")
            raise AudioProcessingError(f"Error in voice interaction: {e}")

    async def _handle_voice_interaction(
        self,
        session: Any,
        voice_input_enabled: bool,
        recording_duration: int
    ) -> None:
        """Handle voice interaction session."""
        
        try:
            while True:
                if voice_input_enabled:
                    prompt = await self._handle_voice_input(recording_duration)
                else:
                    prompt = input("> ")
                                    
                normalized_prompt = ''.join(filter(str.isalnum, prompt.lower()))
                if normalized_prompt in ['exit', 'quit']:
                    print("Exiting conversation")
                    return 
                
                await session.send(input=prompt, end_of_turn=True)
                response_audio = await self._get_voice_output(session, prompt)
                self._transcribe_audio(audio_data = response_audio, speaker="Gemini")
        
        except KeyboardInterrupt:
            logger.info("Voice interaction ended by keyboard interrupt")
            

    async def _handle_voice_input(self, recording_duration: int) -> str:
        """Handle voice input processing."""
        try:
            audio_data = self._record_audio(recording_duration)
            prompt = self._transcribe_audio(audio_data, speaker="User")
            return prompt
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")


    def _delete_files(self, files: List[Any]) -> None:
        """
        Delete uploaded files from the client's storage.
        
        Args:
            files: List of uploaded file objects to delete
            
        Raises:
            FileProcessingError: If file deletion fails
        """
        try:
            for file in files:
                self.client.files.delete(name=file.name)
                logger.info(f"Successfully deleted file: {file.name}")
        except Exception as e:
            logger.error(f"File deletion error: {e}")
            raise FileProcessingError(f"Error deleting files: {e}")

    def _process_url(self, url: str) -> Part:
        """
        Process a YouTube URL into a Part object.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Part: Video part object
            
        Raises:
            VideoProcessingError: If video processing fails
        """
        try:
            video = Part.from_uri(
                file_uri=url,
                mime_type="video/mp4",
            )
            logger.info(f"Successfully processed video URL: {url}")
            return video
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            raise VideoProcessingError(f"Error processing video: {e}")

    def _initialize_conversation(
        self,
        initial_prompt: str,
        files: Optional[List[Any]] = None,
        url: Optional[Any] = None
    ) -> None:
        """
        Initialize a conversation with provided content.
        
        Args:
            initial_prompt: Starting prompt for conversation
            files: Optional list of processed files
            url: Optional processed video URL
            
        Raises:
            GeminiConfigError: If chat creation or message sending fails
        """
        try:
            messages = [initial_prompt]
            if files:
                messages.extend(files)
            if url:
                messages.append(url)

            self.chat = self.client.chats.create(
                model=self.model,
                config=self.content_config,
            )
            logger.info("Chat session created successfully")
            logger.info("Starting conversation with Gemini...")
            for chunk in self.chat.send_message_stream(message=messages):
                if chunk.text:
                    self.console.print(Markdown(chunk.text), end="", no_wrap=True)
                    
        except Exception as e:
            logger.error(f"Chat initialization error: {e}")
            if files:
                self._delete_files(files)
            raise GeminiConfigError(f"Error creating chat: {e}")
            
            
            
    def _setup_chat_loop(self) -> None:
        """
        Set up an interactive chat loop for user conversation.
        
        Raises:
            GeminiConfigError: If chat loop encounters an error
        """
        try:
            while True:
                prompt = input("> ")
                if prompt.lower() in ('exit', 'quit'):
                    logger.info("Chat ended by user")
                    break
                 
                logger.info("Waiting for gemini's response...")    
                for chunk in self.chat.send_message_stream(message=[prompt]):
                    if chunk.text:
                        self.console.print(Markdown(chunk.text), end="", no_wrap=True)
                        
        except KeyboardInterrupt:
            logger.info("Chat ended by keyboard interrupt")
        except Exception as e:
            logger.error(f"Chat loop error: {e}")
            raise GeminiConfigError(f"Error in chat loop: {e}")
        finally:
            print("Chat ended successfully!")

    @staticmethod
    async def _get_voice_output(session: Any, prompt: str) -> ndarray:
        """
        Get voice output from the session.
        
        Args:
            session: Active Gemini session
            prompt: Input prompt
            
        Returns:
            ndarray: Concatenated audio data
            
        Raises:
            AudioProcessingError: If voice output generation fails
        """
        try:
            audio_data_buffer = []
            with OutputStream(samplerate=24340, channels=1, dtype='int16') as stream:
                async for message in session.receive():
                    if message.server_content.model_turn:
                        for part in message.server_content.model_turn.parts:
                            if part.inline_data:
                                audio_data = frombuffer(part.inline_data.data, dtype='int16')
                                stream.write(audio_data)
                                audio_data_buffer.append(audio_data)
                                
            return concatenate(audio_data_buffer)
        except Exception as e:
            logger.error(f"Voice output error: {e}")
            raise AudioProcessingError(f"Error generating voice output: {e}")

    def _record_audio(self, duration: int, fs: int = 24340) -> ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            fs: Sampling frequency in Hz
            
        Returns:
            ndarray: Recorded audio data
            
        Raises:
            AudioProcessingError: If recording fails
        """
        try:
            with InputStream(samplerate=fs, channels=1) as stream:
                logger.info("Starting audio recording...")
                audio = stream.read(int(duration * fs))[0]
                logger.info("Audio recording completed")
                return audio
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            raise AudioProcessingError(f"Error recording audio: {e}")

    def _transcribe_audio(self, audio_data: str, speaker: str = '') -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            str: Transcribed text
            
        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            audio_data = tensor(audio_data.flatten(), dtype = float32).to(self.device_)
            transcription = self.whisper_model.transcribe(audio_data)
            text = transcription.get('text', '')
            logger.info("Audio transcription completed successfully")
            self.console.print(f"{speaker}: {text}")
            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise TranscriptionError(f"Error transcribing audio: {e}")
        
    