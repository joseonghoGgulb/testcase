import os
from flask import Flask, request, jsonify
import covid_detecter

app = Flask(__name__)
myAI = covid_detecter.covid_detecter()

@app.route('/', methods=['POST'])
def hello_world():
    audio = request.files['audio']
    audio.save(os.path.join('./uploads/input', audio.filename))

    out = myAI.detect_covid(audio.filename)

    return jsonify({'probability': out[0][0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
