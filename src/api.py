from flask import Flask

app = Flask("face_detector_app")

@app.route("/")
def home(): 
    return "Face detector working well"

app.run()

