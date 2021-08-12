import random
import os
from flask import Flask, request, jsonify
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

app = Flask(__name__)

@app.route("/predict", methods = ["POST"])

def featurize(filename):
    """ For a given audio clip, calculate the corresponding feature
    Params:
        audio_clip (str): Path to the audio clip
    """
    (rate, sig) = wav.read(filename)
    return mfcc(sig, rate, numcep=13)
def predict():
    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)

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
    os.remove(file_name)

    #send back predicted text
    data = {"words": predicted_sentence}

    # return data
    return jsonify(data)


if __name__==__main__:
    app.run()