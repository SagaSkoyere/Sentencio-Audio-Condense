# Sentencio Audio Condenser

A simple tool to **trim audio/video files down to just the spoken parts**. Designed for language learners to quickly trim episodes in bulk down to sections containing speech -- but also potentially useful for meetings, lectures, podcasts, or any recordings where you only care about the voiced portions. This tool is optimized for minimal setup and ease of use as a portable .exe file. Just download the release and you're done!

---

## Features
- :hourglass: Quick & easy to use! Install and the user-workflow for creation takes less than a minute.
- :brain: **Neural voice detection** (via Silero VAD) for highly accurate accurate speech vs. non-speech identification.
- :scissors: Trims down to **only speech** plus a brief buffer on either side of detected speech to preserve natural sentence flow and dialogue pacing.
- :open_file_folder: **Bulk process entire folders** of audio/video files at once with a single command.
- :arrows_counterclockwise: Extracts audio from common formats (`.wav`, `.mp3`, `.mp4`, `.m4a`, `.mov`, `.mkv`, etc.) via `ffmpeg`.
- :headphones: Makes audio adjustments for easier listening (minor dynamic range compression to make shows or lectures easier to purely listen to). 
- :computer: Works fully offline — no internet connection or API keys rquired. Will always be free to use and open-source.

---

## Install Instructions
1. Download the latest release (in the sidebar on this GitHub).  
2. Extract the ZIP folder containing the program anywhere on your PC. This will take awhile (it's a ~1GB file) but there is no other setup to do. When this completes you're done!

## Run Instructions
1. Run the `.exe` found in the folder you unzipped above whenever you want to create condensed audio files.
2. Specify the folder containing your input file(s) and the folder you want your output files to be sent. Note, both need to be a __folder__ filepath containing your file(s) rather than pointing it at a single audio/video file.

In my testing, a 20-minute video clip takes around 15 seconds to process on a budget laptop. The program will update you as it completes each file and present a final message when done processing the full folder. 

---

## Contact Me (if needed)
- I am not regularly checking GitHub, but if you want to make any updates or get in touch, ping me (Saga) on [Discord](https://discord.gg/85zc78aHwy).

---

## How It Works
1. The Silero VAD neural model assigns a speech probability to each chunk of audio. [Read more on this model here](https://github.com/snakers4/silero-vad).
2. Sections above a set probability threshold are marked by the voice-detection as likely containing speech.  
3. A forward/backward sweep adds a 2-second buffer before & after each identified speech section to preserve natural pacing of spoken dialogue.
4. These remaining audio clips are then exported into the final condensed/trimmed audio output file.

---

## Potential Improvements
- Support output formats beyond `.wav`.  
- Explore masking --> trimmed output for higher exported audio quality.  
- Option to always trim beginning and ending of each file by a hardcoded duration (intro/outro songs).
- Gracefully handle pointing at a single file (instead of a folder).

