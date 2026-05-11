# Sam2 Auto Annotation for YOLO
This is a sam2 based yolo auto annotation tool for drone detection.
It mainly uses ultralytics pretrained model, so it's quite light-weighted for use.


## 1.Create virtual environment

```bash
python -m pip install -r requirements.txt
```

## 2.Prepare your dataset
**If your hardware is good enough to run SAM2 inference locally, you can skip this step.**

Upload your video data to `/video_data/<video_name>/<video_name>.mp4`, and then run :
```bash
python scripts/extract_mask.py -v <video_name>
```
The frames will be extracted in directory `/video_data/<video_name>/rgb_frames/`

If your hardware is good enough to run SAM2 inference locally, you can skip this step.

## 3.Auto annotation
First, prepare a `prompt.yaml` right in your dataset folder, which is used to specify the prompt for sam2.
The format includes 2 parts, you can find it in the folder:
- `initial box prompt `: the bounding box prompt for sam2, in format `xyxy`, the origin is on top-left, and x is horizontal, y is vertical.
 This updates the initial memory of sam2, so it should be precise enough
- `auxiliary prompt`: the auxiliary prompt for sam2, which is used to update the memory of sam2, and it can be a point prompt or a box prompt. The format is `xy` for point prompt, and `xyxy` for box prompt.
This is used to guide the sam2 when it fails in adjacent frames. <font color="red">Note that adding too much prompt will slow down the segmentation.</font>

Then run the following command to start auto annotation:
```bash
python scripts/sam_auto_annotation.py -v <video_name> -i -e
```
After that, download all the masks in `video_data/<video_name>/mask_frames/` if you run sam2 remotely,
then run the following command to convert the masks to yolo format labels:
```bash
python scripts/gen_labels.py -v <video_name>
```

All the labels will be generated in `video_data/<video_name>/labels`, and you can review all the labels in `video_data/<video_name>/check_labels`.
These are images with the labels drawn on them, and you can check if the labels are correct. If not, you can modify the `prompt.yaml` and run the auto annotation again to update the labels.
You can also remove the wrong labels inside it, and then run 
```bash
python scripts/percolate.py -v <video_name>
```
This operation will directly add the correct labels and rgb frames into the yolo train dataset with correct format.



## 4.Train your model
Run the following command to start training:
```bash
python scripts/train.py --epochs 100
```

This is just a simple project that I created it to generate labels for YOLO26n efficiently,
I need it to detect fast and maneuverable drones, so I need lots of data.
It is quite annoying to label the data manually, so I created this tool to help me generate labels automatically.
I have to admit that the quality of the labels is not very good, but it is good enough for my use case.
I hope this tool can help you to generate labels for your own use case, and you can do some improvements on it to make it better.
