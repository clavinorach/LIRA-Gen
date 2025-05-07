# LIRA-Gen
Lipreading Information Resource Assembler-Generator
LIRA-Gen is an end-to-end pipeline tool for generating lipreading datasets (one-second-per-word video) from YouTube videos with Creative Commons BY licenses.

## Overview

LIRA-Gen simplifies the process of creating visual speech recognition (lipreading) datasets from YouTube playlists. The tool handles everything from video downloading to face detection and word-level segmentation, producing standardized clips focused on lip movements for individual words, making it ideal for training and evaluating lipreading models.

## Features

- Automated YouTube playlist processing
- Forced alignment for precise word timing extraction
- Word filtering based on frequency and language dictionary
- Shot detection and scene segmentation
- Face detection and cropping
- One-second clip generation centered on specific words
- CSV metadata generation for dataset management
- Multi-stage pipeline with restart capability

## Prerequisites

- Python 3.10+
- FFmpeg
- Montreal Forced Aligner (MFA)
- PyTorch (with GPU support recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LIRA-Gen.git
cd LIRA-Gen

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with GPU support
# Please follow the official PyTorch installation instructions at https://pytorch.org/get-started/locally/
# We strongly recommend installing PyTorch with GPU support for optimal performance
```

## Directory Structure

LIRA-Gen uses the following directory structure:

```
.
├── video
│   ├── input                  # Video input
│   ├── inputalign             # Forced alignment results
│   ├── inputsubt              # Transcription input
│   ├── output_align_input     # Output for input of forced alignment process
│   ├── output_crop            # Output cropping video
│   ├── output_shot            # Output shot predictions
│   ├── output_shot_info_video # Output sentence per shot
│   ├── output_shot_video      # Output shot trimming video
│   └── output_trim_align      # Output trimming video based on alignment result
```

## Usage

### Basic Usage

```bash
# Run the complete pipeline starting from stage 1
python start.py --videoPlaylist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" --lang "id" --wordFrequency 10 --dictionary "indonesian_words.txt"
```

### Command Line Arguments

```
--videoPlaylist     YouTube playlist link (default: "off" to use local files)
--wordFrequency     Minimum frequency of words to include (default: 10)
--lang              Language code (default: "id" for Indonesian)
--dictionary        Path to language dictionary file (default: "indonesian_words.txt")
--stage             Starting processing stage (default: 1)
```

### Pipeline Stages

LIRA-Gen operates in 8 stages:

1. **Preparing data**: Two options are available:
   - **Option 1 - YouTube Playlist**: Provide a YouTube playlist link with the `--videoPlaylist` parameter, and videos will be automatically downloaded
   - **Option 2 - Local Files**: Place video files (e.g., in.mp4) in `video/input` and corresponding subtitle files (e.g., in.txt) in `video/inputsubt`

2. **Sentence processing**: Extracts transcriptions and prepares for alignment
3. **Align processing**: Prepares files for forced alignment
4. **Word filtering**: Filters words based on frequency and dictionary
5. **Word trimming**: Trims videos based on alignment results
6. **Face cropping**: Detects and crops faces from video frames
7. **CSV data**: Generates metadata CSV files
8. **Face recognition**: Final processing for word-level video clips

## Required Dependencies

LIRA-Gen requires the following dependencies (as listed in requirements.txt):

```
face-recognition==1.3.0
ffmpeg-python==0.2.0
moviepy==2.1.2
numpy==1.26.4
opencv-python==4.11.0.86
pandas==2.2.3
pillow==10.4.0
python_speech_features==0.6
pytorch  # Install from official website with GPU support
scenedetect==0.6.6
scikit-learn==1.6.1
scipy==1.15.2
transnetv2==1.0.0
tensorflow==2.19.0
validators==0.35.0
yt-dlp==2025.4.30
pydub==0.25.1
ffmpeg==1.4
```

**Note:** For PyTorch, we strongly recommend installing with GPU support following the official installation instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for optimal performance.

## Interactive Usage

LIRA-Gen is designed to be interactive, asking for confirmation between certain stages:

1. The system will prompt you to confirm before starting the pipeline
2. After stage 3, it will pause and ask you to run the forced alignment externally
3. You'll need to confirm once alignment is complete to proceed with remaining stages

## Citing LIRA-Gen

If you use LIRA-Gen in your research, please cite:

```
@article{rahmatullah2025recognizing,
  title={Recognizing Indonesian words based on visual cues of lip movement using deep learning},
  author={Rahmatullah, Griffani Megiyanto and Ruan, Shanq-Jang and Li, Lieber Po-Hung},
  journal={Measurement},
  volume={250},
  pages={116968},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgments

- This tool respects YouTube's Terms of Service and only processes videos with appropriate Creative Commons licenses
- Montreal Forced Aligner (MFA) for precise word timing extraction
- Thanks to the open-source computer vision and speech processing communities
