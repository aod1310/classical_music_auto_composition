## Genre Classifier by CNN, song_classifier.h5

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.optimizers import SGD
import numpy as np
import librosa
from fluidsynth.midi2audio import FluidSynth

class GenreClassifier():
    def __init__(self, filename):
        self.filename = filename
        self.idx2genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


    def midi2wav(self):
        fs = FluidSynth('values/Fluid.sf2')
        filepath = 'temp/'
        fs.midi_to_audio('result/generated_music.mid', filepath+self.filename)
        print('midi2wav is done')


    def loadMusic(self):
        if self.filename.endswith('.wav'):
            y, sr = librosa.load('temp/'+self.filename)
            melspec = librosa.feature.melspectrogram(y, sr=sr).T[:1280, ] # 1280 = 30seconds
            musicShape = np.reshape(melspec, (1, melspec.shape[0], melspec.shape[1]))
            return melspec, musicShape
        return None

    def createCNN_Model(self, input_shape):
        wavInput = Input(shape=(input_shape, 128))
        x=wavInput
        levels=64

        for level in range(3):
            x = Conv1D(levels, 3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2, strides=2)(x)
            levels*=2

        x = GlobalMaxPooling1D()(x)

        for fc in range(2):
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)

        labels = Dense(10, activation='softmax')(x)

        model = Model([wavInput], [labels])
        sgd = SGD(lr=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def songClassify(self):
        melspec, music_input = self.loadMusic()

        model = self.createCNN_Model(melspec.shape[0])
        model.load_weights('./values/song_classify.h5')
        prediction = model.predict(music_input)
        predictionidx = np.argmax(prediction)
        res = self.idx2genres[predictionidx]
        percentage = prediction[0][predictionidx] * 100

        return (res, percentage)

if __name__ == '__main__':
    #filepath = './prj1/waves'
    #sc = GenreClassifier(filepath+'/1029-1CONV1D+LSTM.mid.wav')
    #res, pred = sc.songClassify()
    #print(res, pred)
    gc = GenreClassifier('generated_music.wav')
    gc.midi2wav()
