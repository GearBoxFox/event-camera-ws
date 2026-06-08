# event-camera-ws
A workspace containing the needed packages to work with any Prophesee or third-party branded event based camera in ROS2. Packages to work with the Inivation dvXplore cameras will also be added. Also includes tools to do intrensic calibration of event cameras.

## Dependancies

- Metavision SDK 4.2 or greater
- ROS2 Humble
- Python 3.7 or greater


## Usage

#### Cablibrate (Metavision SDK)
1. Collect a `.raw` recording file by running `python3 ext-src/recorder.py` in your desired directory.
2. Convert the `.raw` recording into a `.dat` recording using `metavision_file_to_dat -i file.raw`
3. Convert the `.dat` recording into the older `.h5` recording using `python3 /etx-src/e2calib/python/convert.py`
4. Follow the [e2calib's guide](https://github.com/uzh-rpg/e2calib#image-reconstruction) to create the appropriate `conda` environment
5. Run `e2calib` to generate your calibration dataset using `  python etx-src/e2calib/python/offline_reconstruction.py  --h5file file --freq_hz 5 --upsample_rate 4 --height 480 --width 640`
6. Run `conda deactivate`
7. Run the calibration script using `python3 etx-src/calibration.py` in the same directory as your calibration dataset
