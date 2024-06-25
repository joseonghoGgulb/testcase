import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from keras.models import load_model
from keras import backend as K


class covid_detecter:
    def __init__(self) -> None:
        self.__input_path = './uploads/input'
        self.__img_path = './uploads/img'
        self.__waves_path = './uploads/sound'
        model_path = './model/CNN_dep4_model_best_weights_15.hdf5'
        self.__model = load_model(model_path)
        pass

    def __remove_silence(self, fileName):

        # 30db이하가 0.3초이상 지속
        min_silence_len = 300
        silence_thresh = -30

        # 오디오 파일 불러오기
        audio = AudioSegment.from_file(os.path.join(
            self.__input_path, fileName), format='wav')

        # silence 기준으로 오디오 분할
        chunks = split_on_silence(
            audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

        # silence가 제거된 오디오 조각들을 하나로 합치기
        output = AudioSegment.empty()
        for chunk in chunks:
            output += chunk

        # WAV 파일로 저장
        output.export(os.path.join(self.__waves_path, fileName), format='wav')

    # 오디오 특성 mel_spectrogram

    def __feature_img(self, fileName):
        meanSignalLength = 156027
        signal, sr = librosa.load(os.path.join(self.__waves_path, fileName))
        s_len = len(signal)
        # Add zero padding to the signal if less than 156027 (~4.07 seconds) / Remove from begining and the end if signal length is greater than 156027 (~4.07 seconds)
        if s_len < meanSignalLength:
            pad_len = meanSignalLength - s_len
            pad_rem = pad_len % 2
            pad_len //= 2
            signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                            'constant', constant_values=0)
        else:
            pad_len = s_len - meanSignalLength
            pad_len //= 2
            signal = signal[pad_len:pad_len + meanSignalLength]

        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_mels=128, hop_length=512, fmax=8000, n_fft=512, center=True)
        dbscale_mel_spectrogram = librosa.power_to_db(
            mel_spectrogram, ref=np.max, top_db=80)
        img = plt.imshow(dbscale_mel_spectrogram,
                         interpolation='nearest', origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(self.__img_path, fileName + ".png"),
                    bbox_inches='tight')
        plt.close('all')

    def __loadImages(self, fileName):
        # Loading Images
        images = []
        imgArraySize = (88, 39)

        img = cv2.imread(os.path.join(self.__img_path, fileName+'.png'))
        img = cv2.resize(img, imgArraySize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32)
        img = img/255.0

        # return img

        if img is not None:
            images.append(img)

        return img

    def __AI(self, images):

        images = np.squeeze(images)

        # reshape img
        rows = images.shape[0]
        cols = images.shape[1]

        if K.image_data_format() == 'channels_first':
            images = images.reshape(1, 3, rows, cols)
            input_shape = (3, rows, cols)
        else:
            images = images.reshape(1, rows, cols, 3)
            input_shape = (rows, cols, 3)

        covPredict = self.__model.predict(images)

        return covPredict

    def __rounedValue(self, value):
        decimal_places = 6
        rounded_value = np.round(
            value * 10**decimal_places) / 10**decimal_places
        # formatted_value = "{:.{}f}".format(rounded_value, decimal_places)

        return rounded_value

    def detect_covid(self, fileName: str):
        self.__remove_silence(fileName=fileName)
        self.__feature_img(fileName=fileName)
        img = self.__loadImages(fileName=fileName)
        result = self.__AI(images=img)
        output = self.__rounedValue(value=result)

        return output


if __name__ == '__main__':
    myAI = covid_detecter()
    print(myAI.detect_covid('cough.wav'))
    print(myAI.detect_covid('not_cough.wav'))
