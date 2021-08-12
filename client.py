import requests

URL
TEST_AUDIO_FILEPATH


if __name__==__main__:

    audio_file = open(TEST_AUDIO_FILEPATH,'rb')
    values = {'file':(TEST_AUDIO_FILEPATH,audio_file,"audio/wav")}
    response  = requests.post(URL, files = values)
    data = response.json()

    print(f"Predicted words are: {data['words']}")
