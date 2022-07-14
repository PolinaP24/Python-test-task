from flask import Flask, render_template, request
import cv2
import base64
import numpy as np


web = Flask(__name__)


@web.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        recognize = request.form['img'].split(',')[1]

        np_data = np.frombuffer(base64.b64decode(recognize), np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade_db = cv2.CascadeClassifier('faces.xml')
        faces = face_cascade_db.detectMultiScale(img_gray, 1.3, 2)
        kolvo_faces = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            kolvo_faces += 1

        cv2.imwrite("static/images/p1.jpg", img)

        return render_template("result.html", kolvo_faces=kolvo_faces)
    else:
        return render_template("main.html")


if __name__ == "__main__":
    web.run(debug=True)