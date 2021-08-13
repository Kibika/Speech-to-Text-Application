import random
import os
from flask import Flask, request, jsonify,url_for
import speech_recognition as sr
import pyaudio
import json
import numpy as np
import random
from python_speech_features import mfcc
import librosa
import scipy.io.wavfile as wav
import pickle
from IPython.display import Audio
from IPython.display import Markdown, display
import numpy as np
from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence,int_sequence_to_text
# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model
import webbrowser as wb
from werkzeug.utils import secure_filename


def featurize(file_name):
    """ For a given audio clip, calculate the corresponding feature
    Params:
        audio_clip (str): Path to the audio clip
    """
    (rate, sig) = wav.read(file_name)
    return mfcc(sig, rate, numcep=13)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = 'Upload/'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        audio_file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predict', methods = ["POST"])

def predict():
    # get audio file and save it
#     audio_file = request.files["file"]
#     file_name = str(random.randint(0,100000))
# #     audio_file.save(file_name)
    filename = secure_filename(request.args.get('filename'))
    try:
        if filename and allowed_filename(filename):
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as f:
                return f.read()
    except IOError:
        pass
    return "Unable to read file"
  

    # get model
    model_end = final_model(input_dim=13, units=200)
    model = model_end
    model.load_weights('results/model_end.h5')

    #make a prediction
    data_point = featurize(filename)
    output_length = [model.output_length(data_point.shape[0])]
    prediction = model.predict(np.expand_dims(data_point, axis=0))
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length, greedy=False)[0][0])+1).flatten().tolist()
    predicted_sentence = ''.join(int_sequence_to_text(pred_ints)).replace("<SPACE>", " ")

    #remove audio file
#     os.remove(file_name)

    #send back predicted text
    data = {"words": predicted_sentence}

    # return data
    return jsonify(data)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=False)