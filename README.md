
# Face & Eye Detection using OpenCV (Haar Cascades)

This project demonstrates real-time **face and eye detection** using Haar Cascade classifiers from OpenCV. It includes standalone scripts for detecting faces, eyes, and a combined detector for both — with bounding boxes drawn around the detected regions.

---

##  Files Included

| File Name                      | Description |
|-------------------------------|-------------|
| `Face-Detection.py`           | Detects faces in a webcam stream using `haarcascade_frontalface_default.xml`. |
| `Eye-Detection.py`            | Detects eyes from the webcam using `haarcascade_eye.xml`. |
| `face_eye_detection_combined.py` | Combines both face and eye detection into a single real-time script. |

---

##  What You’ll Learn

- How to use **Haar Cascades** for object detection
- How to use OpenCV with Python for real-time computer vision
- Drawing bounding boxes and labeling detected objects
- Basic structure of a computer vision detection pipeline

---

##  Tools & Libraries

- Python 3.x
- OpenCV (cv2)

Install OpenCV if you haven't already:

```bash
pip install opencv-python
