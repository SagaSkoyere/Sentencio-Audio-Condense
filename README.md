# Sentencio Audio Condenser

A simple tool to **trim audio/video files down to just the spoken parts**. Originally designed for language learners to quickly trim episodes in bulk down to sections containing speech -- but also potentially useful for meetings, lectures, podcasts, or any recordings where you only care about the voices. This tool is optimized for minimal setup and ease of use as a portable .exe file. Just unzip the release .zip and you're done!

---

## Features
- :brain: **Neural voice detection** (Silero VAD) for highly accurate accurate speech vs. non-speech identification.
- :scissors: Trims down to **only speech +1 sec buffer** on either side to preserve sentence flow and dialogue pace.
- :open_file_folder: **Bulk process entire folders** of audio/video files at once with a single command.
- :arrows_counterclockwise: Extracts audio from common formats (`.wav`, `.mp3`, `.mp4`, `.m4a`, `.mov`, `.mkv`, etc.) via `ffmpeg`.  
- :computer: Works fully offline — no internet connection, API keys, or fiddling with lining up subtitle tracks. Will always be free to use and open-source.

---

## Install Instructions
1. Download the latest release (in the sidebar on GitHub).  
2. Extract the ZIP contents anywhere on your PC -- this will take awhile (it's a ~1GB file) but there is no other setup to do. When this completes you're done :)

## Run Instructions
1. Run the `.exe` above whenever you want to bulk process files.
2. Specify the __folder__ containing your input file(s) and the __folder__ you want your output files to be sent. Note, both need to be a folder filepath rather than pointing it at a file.
3. Sit back and relax. In my testing, a 20-minute video clip takes around 30 seconds to process. The program will update you as it completes each file and present a final message when done parsing the full folder. 

---

## How It Works
1. The Silero VAD neural model assigns a speech probability to each ~32 ms chunk. [Read more on this model](https://github.com/snakers4/silero-vad)
2. Chunks above the threshold (0.65) are marked as containing speech.  
3. A two-pass sweep adds a 1-second buffer before and after each identified speech chunk to preserve natural pacing of speech.
4. These remaining chunks are stitched into the final condensed audio output file.

---

## Future Plans
- Support output formats beyond `.wav`.  
- Explore masking --> trimmed output for higher audio quality.  
- User-defined settings: buffer duration (default 1s), speech threshold probability override.  
- Option to trim intros/outros of a hardcoded duration (intros/outros).
- Gracefully handle pointing at a single file (instead of a folder).

---

## Contact Me
- I am not regularly checking GitHub, but if you want to make any updates or get in touch, ping me (Saga) on [Discord](https://discord.gg/85zc78aHwy).
