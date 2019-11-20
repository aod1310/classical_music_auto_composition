## 자동작곡 using CNN-LSTM  CNN-LSTM.h5

import pickle
import numpy as np
import music21
from keras.models import Model
from keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Dense, concatenate
from keras.optimizers import Adam

class AutoCompose():
    def __init__(self):
        self.valuesPath = './values/'
        self.pitchnames = None
        self.lengthnames = None
        self.pitch2int = None
        self.length2int = None
        self.int2pitch = None
        self.int2length = None
        self.start = None


    def loadValues(self):
        pitchesName = 'notepitches.values'
        lengthsName = 'notelengths.values'
        pitchesPath = self.valuesPath+pitchesName
        lengthsPath = self.valuesPath+lengthsName

        # load pitches
        with open(pitchesPath, 'rb') as values:
            _pitches = pickle.load(values)
        # load lengths
        with open(lengthsPath, 'rb') as values:
            _lengths = pickle.load(values)

        lenPitches = len(set(_pitches))
        lenLengths = len(set(_lengths))

        self.pitchnames =  sorted(set(v for v in _pitches))
        self.lengthnames = sorted(set(v for v in _lengths))

        self.pitch2int = {pitch:i for i, pitch in enumerate(self.pitchnames)}
        self.length2int = {str(length):i for i ,length in enumerate(self.lengthnames)}
        self.int2pitch = {i:pitch for i, pitch in enumerate(self.pitchnames)}
        self.int2length = {i:length for i, length in enumerate(self.lengthnames)}

        start = np.random.randint(0, len(_pitches)-51)
        s_pitches = _pitches[start:start+50]
        s_lengths = _lengths[start:start+50]
        self.start = {'pitch':s_pitches, 'length':s_lengths}

    def loadMusicSheet(self, path='temp/entered.mid'):
        enteredPitches = []
        enteredLengths = []
        totalEnteredLength = None
        score = music21.converter.parse(path)
        #score.show('text')
        # parsing for input data
        for part in score:
            for element in part:
                if isinstance(element, music21.instrument.Piano):
                    enteredPitches.append('end')
                    enteredLengths.append(float(0.))
                if isinstance(element, music21.note.Rest):
                    enteredPitches.append('rest')
                    enteredLengths.append(float(element.quarterLength))
                if isinstance(element, music21.note.Note):
                    enteredPitches.append(str(element.pitch))
                    enteredLengths.append(float(element.quarterLength))
                if isinstance(element, music21.chord.Chord):
                    enteredPitches.append('.'.join(str(n) for n in element.normalOrder))
                    enteredLengths.append(float(element.quarterLength))

        if len(enteredPitches) == len(enteredLengths):  # 확실히 다 파싱되었는지확인
            totalEnteredLength = len(enteredPitches)
        #print(enteredPitches)
        return totalEnteredLength, enteredPitches, enteredLengths

    def findSimilarPitch(self, notes):
        notes = notes.split()
        flag = 0
        for note in notes:
            if note not in self.pitch2int:
                flag = flag + 1
                notes.remove(note)
        if len(notes)==flag: return self.pitch2int['rest']
        else: return self.pitch2int[notes[0]]

    def preprocessEntered(self, totalLength, pitches, lengths):
        pitch_in = [self.pitch2int[note] if note in self.pitch2int else self.findSimilarPitch(note) for note in pitches]
        length_in = [self.length2int[str(length)] if str(length) in self.length2int else 1.0 for length in lengths]

        pitch_in = [self.pitch2int[note] for note in self.start['pitch'][:-totalLength]] + pitch_in
        length_in = [self.length2int[str(length)] for length in self.start['length'][:-totalLength]] + length_in

        return pitch_in, length_in

    def createModel(self):
        notes_InputLayer = Input(shape=(50, 1))
        note_conv1d1 = Conv1D(64, 3, activation='relu')(notes_InputLayer)
        note_maxpooling1 = MaxPooling1D(2)(note_conv1d1)
        note_conv1d2 = Conv1D(64, 3, activation='relu')(note_maxpooling1)
        note_maxpooling2 = MaxPooling1D(2)(note_conv1d2)
        note_conv1d3 = Conv1D(128, 3, activation='relu')(note_maxpooling2)
        note_maxpooling3 = MaxPooling1D(2)(note_conv1d3)

        lengths_InputLayer = Input(shape=(50, 1))
        length_conv1d1 = Conv1D(64, 3, activation='relu')(lengths_InputLayer)
        length_maxpooling1 = MaxPooling1D(2)(length_conv1d1)
        length_conv1d2 = Conv1D(64, 3, activation='relu')(length_maxpooling1)
        length_maxpooling2 = MaxPooling1D(2)(length_conv1d2)
        length_conv1d3 = Conv1D(128, 3, activation='relu')(length_maxpooling2)
        length_maxpooling3 = MaxPooling1D(2)(length_conv1d3)

        concatenated = concatenate([note_maxpooling3, length_maxpooling3], axis=-1)

        lstm1 = LSTM(256, return_sequences=True)(concatenated)
        lstm2 = LSTM(512, return_sequences=True)(lstm1)
        lstm3 = LSTM(512)(lstm2)

        noteOutput = Dense(485, activation='softmax')(lstm3)
        lengthOutput = Dense(85, activation='softmax')(lstm3)

        model = Model([notes_InputLayer, lengths_InputLayer], [noteOutput, lengthOutput])
        model.compile(optimizer=Adam(lr=0.00010000000474974513),
                      loss=['categorical_crossentropy', 'categorical_crossentropy'])
        model.load_weights('values/cnnlstm4composing.h5')

        return model


    def compose(self):
        self.loadValues()
        total_length, pitches, lengths = self.loadMusicSheet()

        network_in_pitches, network_in_lengths = self.preprocessEntered(total_length, pitches, lengths)
        model = self.createModel()

        output = []

        #while True:
        for _ in range(150):
            pitch2input = np.reshape(network_in_pitches, (1, len(network_in_pitches), 1))
            length2input = np.reshape(network_in_lengths, (1, len(network_in_lengths), 1))

            predict = model.predict([pitch2input, length2input], verbose=0)
            gen_pitch, gen_length = np.argmax(predict[0]), np.argmax(predict[1])
            if self.int2pitch[gen_pitch] == 'end':
                break
            output.append([self.int2pitch[gen_pitch], self.int2length[gen_length]])

            network_in_pitches.append(gen_pitch); network_in_lengths.append(gen_length)
            network_in_pitches = network_in_pitches[1:len(network_in_pitches)]
            network_in_lengths = network_in_lengths[1:len(network_in_lengths)]


        musicSheet = music21.stream.Stream()
        for e_pitch, e_length in zip(pitches, lengths):
            if e_pitch == 'rest':
                musicSheet.append(music21.note.Rest(quarterLength=e_length))
            elif ('.' in e_pitch) or e_pitch.isdigit():
                pitches_in_chd = e_pitch.split('.')
                chd=[]
                for cur_note in pitches_in_chd:
                    new_pitch = music21.note.Note(int(cur_note))
                    new_pitch.storedInstrument = music21.instrument.Piano()
                    chd.append(new_pitch)
                new_chord = music21.chord.Chord(chd)
                new_chord.quarterLength = e_length
                musicSheet.append(new_chord)
            else:
                new_pitch = music21.note.Note(e_pitch)
                new_pitch.quarterLength = e_length
                new_pitch.storedInstrument = music21.instrument.Piano()
                musicSheet.append(new_pitch)

        for p in output:
            pitch = p[0]; length = p[1]
            if pitch=='rest':
                musicSheet.append(music21.note.Rest(quarterLength=length))
            elif ('.' in pitch) or pitch.isdigit():
                pitches_in_chd = pitch.split('.')
                chd = []
                for cur_note in pitches_in_chd:
                    new_pitch = music21.note.Note(int(cur_note))
                    new_pitch.storedInstrument = music21.instrument.Piano()
                    chd.append(new_pitch)
                new_chord = music21.chord.Chord(chd)
                new_chord.quarterLength = length
                musicSheet.append(new_chord)
            else:
                new_pitch = music21.note.Note(pitch)
                new_pitch.quarterLength = length
                new_pitch.storedInstrument = music21.instrument.Piano()
                musicSheet.append(new_pitch)

        musicSheet.write('midi', fp='result/generated_music.mid')


# test
if __name__ == '__main__':
    import GenreClassifier
    autocompose = AutoCompose()
    autocompose.compose()
    gc = GenreClassifier.GenreClassifier(filename='generated.wav')
    gc.midi2wav()
    res, percentage = gc.songClassify()

    print(res, percentage)