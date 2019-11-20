import pyaudio, wave
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.animation import FuncAnimation
import numpy as np
import sys, os, time
from PyQt5 import QtCore as qtcore
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sheet as st
import AudioProcessing as ap
import AutoCompose
import GenreClassifier

form_class = uic.loadUiType("./ui.ui")[0]

# timer variable
h = 0
m = 0
ms = 0


class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # audio process module call
        self.audio = ap.AudioProcessing()
        devicelst = []
        self.devicelst = []
        for idx in range(self.audio.pa.get_device_count()):
            devicelst.append(self.audio.pa.get_device_info_by_index(idx))
        for info in devicelst:
            if info['maxInputChannels'] != 0 and info['hostApi'] == 0:
                self.devicelst.append(info)
        del devicelst
        self.cb_initialize()

        # draw wave
        self.fig = plt.figure()
        self.canvas = Canvas(self.fig)
        self.plot = self.fig.add_subplot(111, xlim=(0, 3500), ylim=(0, 800000))
        self.graph, = self.plot.plot([], [])
        self.plot.axes.get_xaxis().set_visible(False)
        self.plot.axes.get_yaxis().set_visible(False)
        self.canvas.draw()
        self.ani = FuncAnimation(self.canvas.figure, self.graph_update, frames=50, interval=20, repeat=True)
        self.QV_plot.addWidget(self.canvas)

        # 스톱워치관련
        self.timer = qtcore.QTimer()
        self.stopwatch_reset()
        self.timer.timeout.connect(self.stopwatch_run)
        # 시그널 연결
        self.pbtn_start.clicked.connect(self.start_recording)
        self.pbtn_stop.clicked.connect(self.stop_recording)
        self.pbtn_save.clicked.connect(self.saveMusicSheet)
        self.pbtn_midi.clicked.connect(self.playMusicSheet)
        self.pbtn_play.clicked.connect(self.play_recoded_wav)
        self.pbtn_pause.clicked.connect(self.pause_recorded_wav)
        self.pbtn_genMusic.clicked.connect(self.generateMusic)
        self.cb_devices.currentIndexChanged.connect(self.cb_changed)
        # ui기본설정
        self.pbtn_play.setEnabled(False)
        self.pbtn_save.setEnabled(False)
        self.pbtn_midi.setEnabled(False)
        self.pbtn_stop.setEnabled(False)
        self.pbtn_pause.setEnabled(False)
        self.pbtn_genMusic.setEnabled(False)
        self.pbtn_start.setStyleSheet('QPushButton {color: red;}')
        # button.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')

    def cb_initialize(self):
        for device in self.devicelst:
            self.cb_devices.addItem(device['name'])

    def cb_changed(self):
        selected = self.cb_devices.currentIndex()
        self.audio.closeStream()
        self.audio.openRecordStream(selected)


    def graph_update(self, sample):
        self.graph.set_data(self.audio.freq, self.audio.spec)
        self.canvas.draw()
        return True

    def graph_init(self):
        return self.graph,

    def start_recording(self):
        print('** recording **')
        self.pbtn_stop.setEnabled(True)
        self.pbtn_start.setEnabled(False)
        self.pbtn_play.setEnabled(False)
        self.pbtn_save.setEnabled(False)
        self.pbtn_midi.setEnabled(False)
        self.pbtn_pause.setEnabled(False)
        self.pbtn_genMusic.setEnabled(False)
        self.cb_devices.setEnabled(False)
        self.stopwatch_reset()
        self.stopwatch_start()

        self.audio.start_stream()

        return self

    def stop_recording(self):
        self.pbtn_stop.setEnabled(False)
        self.pbtn_start.setEnabled(True)
        self.pbtn_play.setEnabled(True)
        self.pbtn_save.setEnabled(True)
        self.pbtn_midi.setEnabled(True)
        self.pbtn_pause.setEnabled(True)
        self.pbtn_genMusic.setEnabled(True)
        self.cb_devices.setEnabled(True)
        #self.stopwatch_reset()
        self.timer.stop()

        self.audio.stop_stream()
        self.musicsheet = st.get_sheet()

        return self

    # 스톱워치 초기화
    def stopwatch_reset(self):
        global m, s, ms
        m = 0
        s = 0
        ms = 0
        self.timer.stop()

        start_time = "{0:02d}:{1:02d}.{2:02d}".format(m, s, ms)
        self.lcd_timer.setDigitCount(len(start_time))
        self.lcd_timer.display(start_time)

    # 스톱워치 시작
    def stopwatch_start(self):
        self.timer.start(10)  # 0.01초단위

    # 스톱워치 구현
    def stopwatch_run(self):
        global m, s, ms

        if ms < 99:
            ms += 1
        else:
            if s < 59:
                ms = 0
                s += 1
            elif s == 59 and m < 4:  # 녹음길이 최대 4분
                m += 1
                s = 0
                ms = 0
            else:
                self.stopwatch_reset()
                self.stop_recording()

        time = "{0:02d}:{1:02d}.{2:02d}".format(m, s, ms)
        self.lcd_timer.setDigitCount(len(time))
        self.lcd_timer.display(time)


    def playCallback(self, in_data, frame_count, time_info, status):
        playable = self.playable.readframes(frame_count)
        if playable==b'':
            self.playable.close()
            self.pbtn_pause.setEnabled(False)
            self.pbtn_play.setEnabled(True)
            self.timer.stop()
        else:
            playable = np.fromstring(playable, np.int16)
            self.audio.sample[0:1024] = playable
            self.audio.spec = np.fft.rfft(self.audio.sample[0:1024])
        return (playable, pyaudio.paContinue)

    def play_recoded_wav(self):
        self.playable = wave.open('temp/temp.wav', 'rb')

        self.playstream = self.audio.openPlayStream(self.playable.getsampwidth(),
                                                    self.playable.getnchannels(),
                                                    self.playable.getframerate(),
                                                    callback=self.playCallback)
        self.stopwatch_reset()
        self.stopwatch_start()
        self.pbtn_pause.setEnabled(True)
        self.pbtn_play.setEnabled(False)

        self.playstream.start_stream()

        return self

    def pause_recorded_wav(self):
        if not self.playstream.is_stopped():
            self.pbtn_pause.setText('PLAY')
            self.playstream.stop_stream()
            self.timer.stop()
        else:
            self.pbtn_pause.setText('PAUSE')
            self.playstream.start_stream()
            self.timer.start()
        return self

    def checkPlayEnds(self):
        while True:
            if self.audio.play_data ==b'':
                self.pbtn_pause.setEnabled(False)
                self.pbtn_play.setEnabled(True)
                self.timer.stop()
                return self

    def saveMusicSheet(self):
        self.musicsheet.write('midi', fp='temp/entered.mid')
        QMessageBox.about(self, 'Done', 'saving is done!')

    def playMusicSheet(self):
        self.musicsheet.show('midi')

    def generateMusic(self):
        autocompose = AutoCompose.AutoCompose()
        autocompose.compose()
        gc = GenreClassifier.GenreClassifier(filename='generated.wav')
        gc.midi2wav()
        res, percentage = gc.songClassify()
        if res=='classical' and percentage >= 90:  # 90% 이상일때만 클래식으로 인정.
            QMessageBox.about(self, 'Result', 'this music is classical'+str(percentage))
        else:
            QMessageBox.about(self, 'Result', 'this music is '+res+str(percentage)+'\n generated things will be removed')
            os.remove('temp/generated.wav')
            os.remove('result/generated_music.midi')
        if os.path.exists('temp/entered.mid'):
            os.remove('temp/entered.mid')


    def closeEvent(self, e):
        if os.path.exists('temp.wav'):
            self.audio.deleteWav()
        else:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
