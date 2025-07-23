# Basler Stereo Vision Project

This project utilizes two Basler network cameras to capture frames and generate a 3D image from the captured frames. The application is designed to facilitate stereo vision processing using the Basler camera system.

## Project Structure

```
basler-stereo-vision
├── src
│   ├── main.py                # Entry point of the application
│   ├── camera
│   │   ├── __init__.py        # Initializes the camera module
│   │   └── basler_camera.py    # Manages Basler camera operations
│   ├── stereo
│   │   ├── __init__.py        # Initializes the stereo processing module
│   │   ├── calibration.py      # Camera calibration functions
│   │   └── depth_estimation.py # Depth estimation functions
│   └── utils
│       ├── __init__.py        # Initializes the utility module
│       └── image_processing.py  # Image processing functions
├── config
│   └── camera_config.json      # Configuration settings for the cameras
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd basler-stereo-vision
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Configure the cameras:**
   Edit the `config/camera_config.json` file to set the IP addresses, resolution, and frame rate for your Basler cameras.

4. **Run the application:**
   Execute the main script to capture frames and generate a 3D image:
   ```
   python src/main.py
   ```

## Usage Example

After running the application, the captured frames will be processed to create a 3D image. The output will be displayed or saved based on the implementation in `src/main.py`.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.