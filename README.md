# expert-pancake

## Advanced Motion Detection Using DBSCAN and K-Means Algorithms

This project leverages DBSCAN and K-Means clustering algorithms for advanced motion detection in video files. The input is a video, and the output is the detected motion frames.

### Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
  - [guardar_frames](#guardar_frames)
  - [suma_imagenes](#suma_imagenes)
  - [marcar_clusters](#marcar_clusters)
- [Example](#example)
- [License](#license)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/RogueBaker01/potential-barnacle.git
    ```
2. Navigate to the project directory:
    ```bash
    cd potential-barnacle
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the project, you can use the `Main.py` script, which offers a menu-driven interface for processing video files and detecting motion.

1. Run the script:
    ```bash
    python Main.py
    ```
2. Follow the on-screen prompts to select options and input paths.

### Functions

#### guardar_frames

This function processes the input video to detect motion and saves the frames with detected motion.

**Parameters:**
- `video_path` (str): Path to the input video file.
- `output_dir` (str): Directory to save the processed frames.
- `time_interval` (int): Time interval between frames to process.

#### suma_imagenes

This function sums the matrices of the images in the specified folder and saves the resulting image.

**Parameters:**
- `folder_path` (str): Path to the folder containing images.
- `output_image_path` (str): Path to save the summed image.

#### marcar_clusters

This function applies DBSCAN or K-Means clustering to the summed image to detect and mark clusters of motion.

**Parameters:**
- `image_path` (str): Path to the summed image.
- `use_dbscan` (bool): If `True`, uses DBSCAN; otherwise, uses K-Means.

### Example

To process a video and detect motion:

1. Save the frames from the video:
    ```python
    guardar_frames('input_video.mp4', 'output_frames', 1)
    ```
2. Sum the matrices of the saved frames:
    ```python
    suma_imagenes('output_frames', 'summed_image.png')
    ```
3. Apply clustering to the summed image:
    ```python
    marcar_clusters('summed_image.png', True)
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
