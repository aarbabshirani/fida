import pathlib
from datetime import time, datetime
import flask
import mtcnn
from flask import json
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import face_recognition

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/face-register/user/<user>', methods=['POST'])
def face_register(user):
    try:
        database[user] = user_dict[user]
        user_dict.clear()
    except Exception as e:
        print(e)
        return app.response_class(
            response=json.dumps({
                "error": "internal error"
            }),
            status=500,
            mimetype='application/json')

    return app.response_class(
        response=json.dumps({
            "info": "face registered"
        }),
        status=200,
        mimetype='application/json')


@app.route('/face-login', methods=['POST'])
def face_login():
    # Open the device at the ID 0
    cap = cv2.VideoCapture(0)
    # Check whether user selected camera is opened successfully.
    if not (cap.isOpened()):
        print('Could not open video device')
        cap.release()
        return app.response_class(
            response=json.dumps({
                "error": "Could not open video device"
            }),
            status=500,
            mimetype='application/json')

    now = datetime.now();
    while (datetime.now() - now).total_seconds() < 5:
        ret, frame = cap.read()
        # detect faces in the image
        faces = detector.detect_faces(frame)
        if len(faces) == 1 and faces[0]['confidence'] > 0.99:
            print(faces[0])
            (success, user) = face_exists(encode_face(frame, faces[0]))
            if success:
                cap.release()
                return app.response_class(
                    response=json.dumps({
                        "authorization": True,
                        "user": user
                    }),
                    status=200,
                    mimetype='application/json')
    cap.release()
    return app.response_class(
        response=json.dumps({
            "authorization": False
        }),
        status=401,
        mimetype='application/json')


@app.route('/face-capture/user/<user>', methods=['GET'])
def face_capture(user):
    try:
        # Open the device at the ID 0
        cap = cv2.VideoCapture(0)
        # Check whether user selected camera is opened successfully.
        if not (cap.isOpened()):
            print('Could not open video device')
            cap.release()
            return app.response_class(
                response=json.dumps({
                    "error": "Could not open video device"
                }),
                status=500,
                mimetype='application/json')

        img_file = "img.jpeg";
        path = pathlib.Path().absolute().joinpath(img_file)
        now = datetime.now();
        found = False;
        face = {}
        frame = {}
        while (datetime.now() - now).total_seconds() < 5:
            ret, frame = cap.read()
            # detect faces in the image
            faces = detector.detect_faces(frame)
            if len(faces) == 1 and faces[0]['confidence'] > 0.99:
                print(faces[0])
                face = faces[0]
                cv2.imwrite(img_file, frame)
                found = True
                break;
        cap.release()

        if found:
            body = {
                "img": str(path)
            }
            user_dict[user] = get_encoded(str(path))
            #user_dict[user] = encode_face(frame, face)
            return app.response_class(
                response=json.dumps(body),
                status=200,
                mimetype='application/json')

    except Exception as e:
        print(e)

    return app.response_class(
        response=json.dumps({
            "error": "couldn't detect one face, make sure there is only one person facing the camera"
        }),
        status=406,
        mimetype='application/json')


def encode_face(frame, face):
    # get coordinates
    x1, y1, width, height = face['box']
    if x1 < 0:
        x1 = 0
    x2, y2 = x1 + width, y1 + height
    # encode faces
    known_face_locations = [(0, width, height, 0)]
    return face_recognition.face_encodings(frame[y1:y2, x1:x2], known_face_locations)


def face_exists(encoded_image):
    min_dist = 100
    identity="unknown"
    for (name, db_enc) in database.items():
        print(name)
        dist = np.linalg.norm(encoded_image-db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.5:
        print("Not in the database.")
        min_dist = 100
        identity = "unknown"
        return False, ""
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return True, str(identity)


def get_encoded(filename):
    # image = face_recognition.load_image_file(fileName)
    image = pyplot.imread(filename)
    faces = detector.detect_faces(image)
    if (len(faces) == 0):
        return
    result = faces[0]
    # for result in faces:
    # get coordinates
    x1, y1, width, height = result['box']
    if x1 < 0:
        x1 = 0
    x2, y2 = x1 + width, y1 + height
    known_face_locations = [(0, width, height, 0)]
    face_encoding = face_recognition.face_encodings(image[y1:y2, x1:x2], known_face_locations)[0]
    return face_encoding


detector = MTCNN()
user_dict = {}
database = {}
#database["Amir"] = get_encoded("C:\\Users\\aarba\\OneDrive\\Pictures\\me.jpg")
app.run()
