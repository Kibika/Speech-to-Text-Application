# speech-to-text
The project uses speech data and their corresponding transcriptions to train a deep learning model that can be used to collect Swahili speech data and store it's corresponding text.

Since speech data cannot be directly input into a deep learning model, the speech data undergoes transformations that turns the speech into an image that can be input into a deep learning model.

# Preprocessing
The speech is resized, mono files are converted to stereo and a common sample rate and duration is set for all the files. The raw audio goes through augmentation to allow resampling. A spectrogram is generated out of the audio. A spectrogram is an image of the speech showing the frequency and amplitude. SpecAugment methods are use to resample the images of the audios. The image can be a spectrogram or MelSpectrogram, the AudioGenerator class allows a user to choose whether to use a spectrogram or MelSpectrogram. A MelSpectrogram gives the model the ability to perceive frequency and amplitude the way humans speak. 
The text is converted to numbers using the characters in char_map.py

# Modelling
Several models are used, the word error rate(wer) is calculated and the bidirctional LSTM is chosen as the best model for prediction. The bidirectional LSTM model is also chosen because of its ability to use past data, future data and information obtained from past data to train the network. Speech and text are a form of sequential data.

# Credits
https://github.com/energyfirefox/DNNSpeechRecognizerAIND

https://github.com/udacity/AIND-VUI-Capstone
