import sys
import argparse
import cv2
from time import time
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder import FaceEncoderModels, FaceEncoder
from libfaceid.liveness import FaceLivenessModels, FaceLiveness


# Use flask for web app
from flask import Flask, render_template, Response, session, request, flash

app = Flask(__name__)

# Set the window name
#WINDOW_NAME = "Facial_Recognition"

# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"
INPUT_DIR_MODEL_LIVENESS        = "models/liveness/"

# Set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)



"""def cam_init(width, height):
    cap = cv2.VideoCapture(cam_index)
    if sys.version_info < (3, 0):
        cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
"""

def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        if confidence is not None:
            text = "{} {:.2f}%".format(face_id, confidence)
        else:
            text = "{}".format(face_id)
        cv2.putText(frame, text, (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close):
    if eyes_close:
        #print("eye less than threshold {:.2f}".format(eyes_ratio))
        eye_counter += 1
    else:
        #print("eye:{:.2f} blinks:{}".format(eyes_ratio, total_eye_blinks))
        if eye_counter >= eye_continuous_close:
            total_eye_blinks += 1
        eye_counter = 0
    return total_eye_blinks, eye_counter


def monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open):
    if mouth_open:
        #print("mouth more than threshold {:.2f}".format(mouth_ratio))
        mouth_counter += 1
    else:
        #print("mouth:{:.2f} opens:{}".format(mouth_ratio, total_mouth_opens))
        if mouth_counter >= mouth_continuous_open:
            total_mouth_opens += 1
        mouth_counter = 0
    return total_mouth_opens, mouth_counter


# process_livenessdetection is supposed to run before process_facerecognition
@app.route('/login', methods=['POST'])
def process_livenessdetection():
    cap = cv2.VideoCapture(0)
    windowname = "Face_Recognition"
    #cam_index = 0
    cam_resolution = RESOLUTION_QVGA
    #detector = FaceDetectorModels.HAARCASCADE
    #    detector=FaceDetectorModels.DLIBHOG
    #    detector=FaceDetectorModels.DLIBCNN
    #    detector=FaceDetectorModels.SSDRESNET
    #    detector=FaceDetectorModels.MTCNN
    detector=FaceDetectorModels.FACENET

    #encoder = FaceEncoderModels.LBPH
    #    encoder=FaceEncoderModels.OPENFACE
    #    encoder=FaceEncoderModels.DLIBRESNET
    encoder=FaceEncoderModels.FACENET

    liveness = FaceLivenessModels.EYESBLINK_MOUTHOPEN
    #liveness=FaceLivenessModels.COLORSPACE_YCRCBLUV

    # Initialize the camera
    #camera = cam_init(cam_resolution[0], cam_resolution[1])

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=detector, path=INPUT_DIR_MODEL_DETECTION)

        # Initialize face recognizer
        face_encoder = FaceEncoder(model=encoder, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        # Initialize face liveness detection
        face_liveness  = FaceLiveness(model=liveness, path=INPUT_DIR_MODEL_LIVENESS)
        face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)

    except:
        print("Error, check if models and trained dataset models exists!")
        return

    face_id, confidence = (None, 0)

    eyes_close, eyes_ratio = (False, 0)
    total_eye_blinks, eye_counter, eye_continuous_close = (0, 0, 1) # eye_continuous_close should depend on frame rate
    mouth_open, mouth_ratio = (False, 0)
    total_mouth_opens, mouth_counter, mouth_continuous_open = (0, 0, 1) # eye_continuous_close should depend on frame rate

    time_start = time()
    time_elapsed = 0
    frame_count = 0
    identified_unique_faces = {} # dictionary
    runtime = 60 # monitor for 10 seconds only
    is_fake_count_print = 0


    print("Note: this will run for {} seconds only".format(runtime))
    while (True):
        # Capture frame from webcam
        ret, frame = cap.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break


        # Detect and identify faces in the frame
        # Indentify face based on trained dataset (note: should run facial_recognition_training.py)
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):

            # Check if eyes are close and if mouth is open
            eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
            mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)
            print("eyes_close={}, eyes_ratio ={:.2f}".format(mouth_open, mouth_ratio))
            print("mouth_open={}, mouth_ratio={:.2f}".format(mouth_open, mouth_ratio))

            # Detect if frame is a print attack or replay attack based on colorspace
            is_fake_print  = face_liveness2.is_fake(frame, face)
            #is_fake_replay = face_liveness2.is_fake(frame, face, flag=1)

            # Identify face only if it is not fake and eyes are open and mouth is close
            if is_fake_print:
                is_fake_count_print += 1
                face_id, confidence = ("Fake", None)
            elif not eyes_close and not mouth_open:
                face_id, confidence = face_encoder.identify(frame, face)
                if face_id not in identified_unique_faces:
                    identified_unique_faces[face_id] = 1
                else:
                    identified_unique_faces[face_id] += 1

            label_face(frame, face, face_id, confidence)
            cv2.imshow(windowname,frame)
            cv2.waitKey(1)
            # Set text and bounding box on face
            print("face_id :" , face_id)
            print("confidence :", confidence)
            break  # Process 1 face only
        #yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')


        # Monitor eye blinking and mouth opening for liveness detection
        total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close)
        total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open)
        # Update frame count
        frame_count += 1


        if face_id in identified_unique_faces:
            return render_template('success.html')
        else:
            return render_template('result_F.html')
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('test2.html')
    else:
        return "Hello!"

'''@app.route('/login', methods=['POST'])
def login():
     return Response(process_livenessdetection(), mimetype='multipart/x-mixed-replace; boundary=frame')
'''

# Entry point for web app
'''@app.route('/video_viewer')
def video_viewer():
    return Response(process_livenessdetection(), mimetype='multipart/x-mixed-replace; boundary=frame')

'''

if __name__ == '__main__':
    app.run(host='localhost', threaded=True, debug=True, port=1000)
