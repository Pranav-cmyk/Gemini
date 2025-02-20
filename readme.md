# Google Gemini API Integration

This project provides a simple interface for interacting with the Google Gemini API.

## Prerequisites

-   Python 3.11+
-   A Google Cloud project with the Gemini API enabled
-   A Google Cloud API key

## Installation


pip install -r requirements.txt


## Setup

1.  **Enable the Gemini API:**

    -   Go to the [Google Cloud Console](https://console.cloud.google.com/).
    -   Select your project.
    -   Navigate to "APIs & Services" > "Library".
    -   Search for "Gemini API" and enable it.

2.  **Create an API Key:**

    -   In the Google Cloud Console, navigate to "APIs & Services" > "Credentials".
    -   Click "Create credentials" > "API key".
    -   Copy the API key.


## Usage

```python
import os
import google.generativeai as genai
from Gemini.Google_Gemini import Google_Gemini  # Adjust import path if needed
```

# Option 1: Initialize with the API key from the environment variable
```python
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
gemini = GoogleGemini(api_key = api_key)
```

# Option 2: Create a .env file with the API key
In your .env file, add the line: GOOGLE_API_KEY=YOUR_API_KEY after which
the GoogleGemini class will automatically load the API key from the .env file.


# Example 1: Using the conversational_assistant method
```python
from Google_Gemini_main import GoogleGemini, GeminiContentConfig
from google.genai.types import Tool, GoogleSearch
from os import path, listdir


# Tool Code execution is not supported with file or url uploads # 
# but google_search is supported
# Change that and configure additional parameters

search_tool = Tool(
    google_search=GoogleSearch()
)

custom_config = GeminiContentConfig(
    temperature=0.9,
    max_output_tokens=8096,
    top_k=5,
    top_p=0.85,
    system_instruction="You are a professional but friendly assistant",
    tools = [search_tool]
)

#Pass those configs to the content_config parameter of the class
gemini = GoogleGemini(content_config=custom_config)

#helper method to get a list of all files from a directory
def get_files(directory):
    return [path.join(directory, f) for f in listdir(directory)]


gemini.conversational_assistant(
    files = get_files('./Digital'), # some directory
    initial_prompt="""
    
    You are a professional Course Creator and you have been tasked 
    with creating a course based on the files uploaded. Your task is 
    to sort all the information and create a course that is easy 
    to understand and follow. The course should be engaging and 
    informative and must include references from external sources 
    like youtube or google to support the information provided from 
    the files. Your delivery must be professional and engaging.
    
    """
    #youtube_url = "some_url" # replace with your youtube url 
    #and uncomment this line
)
```

# Example 2: Using the start_voice_interaction method
```python
import asyncio
from Gemini.Google_Gemini import Google_Gemini

async def voice_interaction():
    gemini = Google_Gemini()

    try:
        #You can specify the voice from a list of predefined voices
        #You can enable voice_input and the code will listen to your 
        #voice for the specified recording_duration
        await gemini.start_voice_interaction(
            voice = 'Aoede',
            voice_input_enabled = True,
            recording_duration = 8,
            system_instructions = "You are a helpful assistant.",
        )
    except Exception as e:
        print(f"An error occurred: {e}")

asyncio.run(voice_interaction())
```

# Important Notes:
1. Ensure all required packages are installed using `pip install -r requirements.txt`if you run into any issues ensure that you've created a .env file with the variable GOOGLE_API_KEY set to your actual api key.
2. Replace dummy file paths and YouTube URL with actual ones in your implementation
3. The `start_voice_interaction` method is asynchronous and requires proper async handling
4. This method requires python 3.11 or higher to support asyncio.


## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[Apache 2.0 License](LICENSE)
