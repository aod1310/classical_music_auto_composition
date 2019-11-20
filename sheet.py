import librosa, librosa.display
import numpy as np
from music21 import *
import scipy.signal as sg
import peakutils


def read_wav(filename):
    y, sr = librosa.load(filename, sr=22050)
    return y, sr

# 음의 구간 결정
def get_onsetboundaries(y, sr=22050):
    tempo, beat_times = librosa.beat.beat_track(y, sr=sr, start_bpm=60, units='time')
    hop_length = 300
    onset_samples = librosa.onset.onset_detect(y, sr=sr, units='samples', hop_length=hop_length, backtrack=False,
                                               pre_max=15, post_max=8, pre_avg=90, post_avg=90, delta=0.15,
                                               wait=0)
    onset_boundaries = np.concatenate([[0], onset_samples, [len(y)]]) # 첫 음과 끝 음의 길이를 계산하기위해 처음과 끝도 삽입해 합병
    return onset_boundaries

# 원본
'''
def get_pitch(section, sr):
    section = sg.medfilt(section)
    C = np.abs(librosa.cqt(section, sr=sr, norm=1))
    s = []
    for c in C:
        if np.log(sum(c)**2) < 2:
            s.append(0)
        else:
            s.append(sum(c)**2)
    s = np.array(s)
    #print(s)
    indexes = convert_to_pitch(librosa.util.peak_pick(s, pre_max=8, post_max=13, pre_avg=90, post_avg=90, delta=0.2, wait=0))
    return indexes
    '''
# 입력된 음의 pitch(chord) 추출
def get_pitch(section, sr):
    #section = sg.medfilt(section)
    C = np.abs(librosa.cqt(section, sr=sr, norm=1))
    s = []
    for c in C:  # 출력값이 작은(잡음 etc) 부분은 과감하게 삭제 음이 있는곳은 높은 출력값을 나타내게 되어있음
        if np.log(sum(c)**2) < 2:
            s.append(0)
        else:
            s.append(sum(c))
    s = np.array(s)
    s_max = s.max()
    s_min = s.min()  # 어떤 출력값이 들어와도 일정하게 peak를 검출하기위해 정규화
    if s_max == 0: return []
    s = (s-s_min) / (s_max-s_min)
    #indexes = convert_to_pitch(librosa.util.peak_pick(s, pre_max=11, post_max=16, pre_avg=90, post_avg=90, delta=0.2, wait=1))
    #indexes = convert_to_pitch(librosa.util.peak_pick(s, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=1))
    indexes = convert_to_pitch(peakutils.indexes(s, thres=0.8879, thres_abs=True))
    return indexes

#
def convert_to_pitch(chords):
    note = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes = []
    for o in range(1, 8):
        for n in note:
            pitch = n + str(o)
            notes.append(pitch)
    result = []
    for c in chords:
        result.append(notes[int(c)])
    return result


def set_duration(ob):   # 맵핑을 어떻게하지?... 0.5, 0.25 등에 맞출 수 있는..
    duration_able = [(1/2)**n for n in range(0, 10)]
    duration_able.append(2)
    duration_able.append(4)
    times = [2*(ob[point+1]-ob[point]) for point in range(len(ob)-1)]
    near = None
    for i, t in enumerate(times):
        m = 9999
        for t_d in duration_able:
            if abs(t-t_d) < m:
                m = abs(t-t_d)
                near = t_d
        times[i] = near
    return times

def init_stream():
    return stream.Stream()

def set_chrod(notes):
    return chord.Chord(notes)

def set_rest(length):
    return note.Rest(quaterLength=length)
'''
# get_pitch() test
test = []
y, sr = read_wav('./aqe.wav')
onset_boundaries = get_onsetboundaries(read_wav('./aqe.wav'))

for i in range(len(onset_boundaries)-1):
    p0 = onset_boundaries[i]
    p1 = onset_boundaries[i+1]
    pitch = get_pitch(y[p0:p1], sr)
    test.append([p for p in pitch])


# test music sheet
times = set_duration(librosa.samples_to_time(onset_boundaries, sr=sr))
print(times)
music_sheet = stream.Stream()
for notes, length in zip(test, times):
    n = chord.Chord(notes)
    n.quarterLength = length
    music_sheet.append(n)


music_sheet.show()
'''


def get_sheet(filename='temp/temp.wav'):
    y, sr = read_wav(filename)
    onset_boundaries = get_onsetboundaries(y, sr)

    chords = []
    for i in range(len(onset_boundaries) - 1):
        p0 = onset_boundaries[i]
        p1 = onset_boundaries[i + 1]
        pitch = get_pitch(y[p0:p1], sr)
        chords.append([p for p in pitch])

    times = set_duration(librosa.samples_to_time(onset_boundaries, sr=sr))
    music_sheet = init_stream()
    for notes, length in zip(chords, times):
        if notes == []:
            n = set_rest(length)
            music_sheet.append(n)
            continue
        n = set_chrod(notes)
        n.quarterLength = length
        music_sheet.append(n)

    #music_sheet.write('midi', fp='instrument.mid')
    #music_sheet.show('text')
    #music_sheet.write('midi', fp='result2.mid')
    return music_sheet



if __name__ =='__main__':
    get_sheet('doremi.wav')