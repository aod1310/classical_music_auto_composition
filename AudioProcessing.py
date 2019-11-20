import pyaudio
import numpy as np
import wave
import os

class AudioProcessing():
    def __init__(self):
        self.FORMAT= pyaudio.paInt16
        self.CHUNK = 1024
        self.RATE = 44100
        self.CHANNEL = 1
        self.frames=[]
        self.sample = np.zeros([102400])
        self.freq = np.fft.rfftfreq(1024, 1 / self.RATE)
        self.spec = np.zeros([1024], dtype=complex)
        self.pa = pyaudio.PyAudio()
        self.openRecordStream(0)
        self.play_data = None

    def openRecordStream(self, deviceidx):
        self.stream = self.pa.open(format=self.FORMAT,
                                   channels=self.CHANNEL,
                                   rate=self.RATE,
                                   input=True,
                                   output=True,
                                   input_device_index=deviceidx,
                                   stream_callback=self.recordCallback)
        print(self.pa.get_device_info_by_index((deviceidx)))


    def closeStream(self):
        self.stream.close()
        del self.stream

    def recordCallback(self, in_data, frame_count, time_info, flag):
        self.frames.append(in_data)
        in_data = np.fromstring(in_data, np.int16)
        for i in np.arange(99, 0, -1):
            self.sample[i*1024 : (i+1) * 1024] = self.sample[(i-1) * 1024 : i*1024]  # chunk만큼 뒤로 당기고
        self.sample[0:1024] = in_data # 새로운 CHUNK를 맨앞에 입력함
        # fft
        self.spec = np.fft.rfft(self.sample[0:1024])
        return (in_data, pyaudio.paContinue)

    def start_stream(self):
        self.frames=[]
        self.stream.start_stream()
        return self


    def stop_stream(self):
        self.stream.stop_stream()
        wf = wave.open('temp/temp.wav', 'wb')
        wf.setnchannels(self.CHANNEL)
        wf.setsampwidth(self.pa.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print('recording is done......')
        return self

    def openPlayStream(self, format, channels, rate, callback=None):
        #self.playwf = wave.open('temp.wav', 'rb')
        pa = pyaudio.PyAudio()
        playable_stream = pa.open(format=pa.get_format_from_width(format),
                                  channels=channels,
                                  rate=rate,
                                  output=True,
                                  stream_callback=callback)
        return playable_stream

        # 문제있음
    '''
    def playCallback(self, in_data, frame_count, time_info, status):
        self.play_data = self.playwf.readframes(frame_count)

        if self.play_data==b'':
            self.playwf.close()
        else:
            data = np.fromstring(self.play_data, np.int16)
            self.sample[0:1024] = data
            self.spec = np.fft.rfft(self.sample[0:1024])

        return (self.play_data, pyaudio.paContinue)

    '''
    def deleteWav(self):
        if os.path.exists('temp/temp.wav'):
            os.remove('temp/temp.wav')
        else:
            print('cannot found "temp.wav"')


# test
if __name__ == '__main__':
    pa = pyaudio.PyAudio()

    devicelst = []
    for idx in range(pa.get_device_count()):
        devicelst.append(pa.get_device_info_by_index(idx))

    for info in devicelst:
        if info['maxInputChannels'] != 0 and info['hostApi'] == 0:
            print(info)

    #ap = AudioProcessing()
    #ap.openRecordStream()
    #ap.start_stream()
    #time.sleep(5)
    #ap.stop_stream()
    #ap.stream.close()
