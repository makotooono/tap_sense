# -*- coding: utf-8 -*-
# standard 
import sys
import struct
import pickle
# original
from tools import *
from plotter import *  # plot graph
# machine learning
from svm import *
from svmutil import *
# gui
from PySide.QtGui import *
from PySide.QtCore import *
# other
import numpy as np  # vector processing 
import pyaudio as pa  # audio input 
import lightblue as lb  # bluetooth


#滑らかに変化するプログレスバー
class SmoothBar(QProgressBar):
    def __init__(self, parent=None):
        super(SmoothBar, self).__init__(parent)
        self._acc = 1

    @Slot()
    def setValue(self, value):
        if self.value() < value:
            filterd_value = value
            self._acc = 1
        else:
            filterd_value = max(self.minimum(),
                    self.value() - self.maximum() * 0.01 * self._acc)
            self._acc += 0.1
        super(SmoothBar, self).setValue(filterd_value)
        

class MainWindow(QMainWindow):
    _volumeChanged = Signal(int)
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        #ウィンドウタイトルの設定
        self.setWindowTitle("tap_sense_server")

        #入力デバイスの取得
        self._p = pa.PyAudio()
        self._input = None
        n = self._p.get_device_count()
        self._devices = [self._p.get_device_info_by_index(i) for i in range(n)]

        #SVMモデルの初期化
        self._model = None
        self._train_dict = {"tip": [], "pad": [], "nail": [], "knock": []}

        # UI周りの設定
        self._setup_ui()
        self._setup_toolbar()

        #デフォルト動作オプションの設定
        self.set_buffer_size(2)
        self.set_input_device(0)
        self._buffer_combo.setCurrentIndex(2)
        self._input_combo.setCurrentIndex(0)

        #各バッファの初期化
        self._init_buffers()

        #プロセスタイマー始動
        self._timer = QTimer()
        self._timer.timeout.connect(self._process)
        self._timer.start(0)


    def _init_buffers(self):
        empty_data = [0] * self._sample_size
        self._peak_cands = [0] * 3
        self._peak_wave_cands = [empty_data] * 3
        self._peak_spectle = empty_data
        self._peak_wave = empty_data
        self._spectle = empty_data
        self._wave = empty_data

    @Slot(int)
    def set_input_device(self, n):
        if self._input:
            self._input.close()
            
        self._input = self._p.open(
                format=pa.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=n,
                frames_per_buffer=self._sample_size)

        self._init_buffers()

    @Slot(int)
    def set_buffer_size(self, n):
        self._sample_size = 2 ** (9 + n)
        self._init_buffers()
        self._wave_viewer.plot_range = (0, self._sample_size, -1, 1)
        # x軸の生成
        df = 44100 / float(self._sample_size)
        self._xs = [df * i for i in range(0, self._sample_size / 2)]


    #入力波形の取得
    def _get_wave(self):
        try:
            wave_int16 = struct.unpack(
                    str(self._sample_size) + "h",
                    self._input.read(self._sample_size)
                    )
            wave = map(lambda x: x/32768., wave_int16)
        except IOError:
            wave = None
        return wave


    #ピークの検出
    def _update_peak(self, volume):
        n = self._sample_size
        self._peak_cands.pop(0)
        self._peak_wave_cands.pop(0)
        self._peak_cands.append(volume)
        self._peak_wave_cands.append(self._wave)

        p_cands = self._peak_cands
        if p_cands[1] > 0.01 and p_cands[0] < p_cands[1] > p_cands[2]:
            is_peak = True
            pwc = np.array(self._peak_wave_cands)
            connected = pwc.flatten()
            norm = lambda x: x / np.amax(x)
            offs =  int(np.where(norm(pwc[1]) > 0.3)[0][0])
            pre = n / 4
            post = n - pre
            left = n + offs - pre
            right = n + offs + post
            self._peak_wave = norm(connected[left:right]).tolist()
            self._peak_spectle = wave2spectle(self._peak_wave)
        else:
            is_peak = False

        return is_peak

    @Slot()
    def _process(self):
        n = self._sample_size
        #入力信号の取得
        wave = self._get_wave()
        if not wave:  # 波形の取得に失敗したらなにもしない
            return

        #ボリュームを求める
        volume = max(map(lambda x: abs(x), wave))

        #全体波形の更新
        self._wave = self._wave[len(wave):] + wave

        #スペクトルを求める
        self._spectle = wave2spectle(self._wave)

        #タッチの検出
        is_peak = self._update_peak(volume)
        if is_peak:
            debug = self._debug_edit
            debug.clear()

            #学習データの追加
            if self._train_action.isChecked():
                current_label = self._train_combo.currentText()
                self._train_dict[current_label].append(self._peak_spectle)
                #各学習回数を表示
                for i in range(self._train_combo.count()):
                    label = self._train_combo.itemText(i)
                    count = len(self._train_dict[label])
                    debug.append("%s_count:%d" % (label, count))

            #タッチパターンの予測
            if self._predict_action.isChecked() and self._model:
                result = svm_predict([0], [self._peak_spectle], self._model)
                judge = self._train_combo.itemText(int(result[0][0]))
                #予測結果の表示
                debug.append(str(result))
                debug.append(judge)

        
        #表示の更新
        self._volume_bar.setValue(volume * self._volume_bar.maximum())
        self._wave_viewer.remove_all()
        self._spectle_viewer.remove_all()
        self._wave_viewer.add_data(self._wave)
        self._wave_viewer.add_data(self._peak_wave)
        self._spectle_viewer.add_data(zip(self._xs, self._spectle[:n/2]))
        self._spectle_viewer.add_data(zip(self._xs, self._peak_spectle[:n/2]))
        self._wave_viewer.update()
        self._spectle_viewer.update()

    def _setup_ui(self):
        #ボリューム表示用のバー
        self._volume_bar = SmoothBar()
        self._volume_bar.setOrientation(Qt.Vertical)
        self._volume_bar.setRange(0, 100000)
        self._volumeChanged.connect(self._volume_bar.setValue)

        #入力信号ビューワー
        self._wave_viewer = Plotter()
        self._wave_viewer.setFixedHeight(100)

        #スペクトルビューワー
        self._spectle_viewer = Plotter()
        self._spectle_viewer.plot_range = (0, helz2mel(22050), 0, 30)
        self._spectle_viewer.ismel = True

        #学習対象選択コンボボックス
        self._train_combo = QComboBox()
        for key in sorted(self._train_dict.keys()):
            self._train_combo.addItem(key)

        #入力デバイス選択コンボボックス
        self._input_combo = QComboBox()
        for device in self._devices:
            self._input_combo.addItem(device["name"])
        self._input_combo.currentIndexChanged.connect(self.set_input_device)

        #サンプルサイズ選択用コンボボックス
        self._buffer_combo = QComboBox()
        for size in range(5):
            self._buffer_combo.addItem(str(2 ** (9 + size)))
        self._buffer_combo.currentIndexChanged.connect(self.set_buffer_size)

        # bluetooth関連ウィジェット
        self._BT_state = QLabel("None")
        self._BT_select = QPushButton("connect")
        self._BT_select.clicked.connect(self._connect_BT)

        #デバッグ出力用テキストエディット
        self._debug_edit = QTextEdit()
        self._debug_edit.setReadOnly(True)
        self._debug_edit.setMinimumHeight(130)

        #モデルファイル操作メニュー
        self._model_file_edit = QLineEdit()
        self._model_file_edit.setText(QDir.currentPath()+"/untitled.model")
        self._file_select = QPushButton("select")
        self._file_select.clicked.connect(self._select_file)
        self._file_save = QPushButton("save")
        self._file_save.clicked.connect(self._save_model)
        self._file_load = QPushButton("load")
        self._file_load.clicked.connect(self._load_model)

        #モデルファイルレイアウト
        file_lay = QHBoxLayout()
        file_lay.addWidget(QLabel("Model File"))
        file_lay.addWidget(self._model_file_edit)
        file_lay.addWidget(self._file_select)
        file_lay.addWidget(self._file_save)
        file_lay.addWidget(self._file_load)

        #ビューワーレイアウト
        plotter_lay = QVBoxLayout()
        plotter_lay.addWidget(self._wave_viewer)
        plotter_lay.addWidget(self._spectle_viewer)
        plotter_lay.addWidget(self._debug_edit)
        plotter_lay.addLayout(file_lay)

        #全体レイアウト
        lay = QHBoxLayout()
        lay.addLayout(plotter_lay)
        lay.addWidget(self._volume_bar)

        w = QWidget()
        w.setLayout(lay)
        self.setCentralWidget(w)

    def _setup_toolbar(self):
        self._toolbar = self.addToolBar("Tools")

        #学習モードアクション
        self._train_action = QAction("Train", self)
        self._train_action.setCheckable(True)
        self._train_action.triggered.connect(self._train_clicked)
        self._train_action.setChecked(True)

        #予測モードアクション
        self._predict_action = QAction("Predict", self)
        self._predict_action.setCheckable(True)
        self._predict_action.triggered.connect(self._predict_clicked)

        #モード選択メニュー
        self._toolbar.addAction(self._train_action)
        self._toolbar.addAction(self._predict_action)
        self._toolbar.addSeparator()

        #学習リセット・学習対象選択メニュー
        resetButton = QPushButton("reset")
        resetButton.clicked.connect(self._reset_train)
        self._toolbar.addWidget(QLabel("Train Target"))
        self._toolbar.addWidget(self._train_combo)
        self._toolbar.addWidget(resetButton)
        self._toolbar.addSeparator()

        #入力デバイス・サンプルサイズの選択
        self._toolbar.addWidget(QLabel("Input Device"))
        self._toolbar.addWidget(self._input_combo)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(QLabel("Buffer Size"))
        self._toolbar.addWidget(self._buffer_combo)

        #bluetooth関連メニュー
        self._toolbar.addSeparator()
        self._toolbar.addWidget(QLabel("bluetooth:"))
        self._toolbar.addWidget(self._BT_state)
        self._toolbar.addWidget(self._BT_select)

    #学習データのリセット
    @Slot()
    def _reset_train(self):
        self._train_dict = {}
        for i in range(self._train_combo.count()):
            label = self._train_combo.itemText(i)
            self._train_dict[label] = []
        self._debug_edit.clear()

    #学習モードへ切り替え
    @Slot()
    def _train_clicked(self):
        self._train_action.setChecked(True)
        self._predict_action.setChecked(False)

    #予測モードへ切り替え
    @Slot()
    def _predict_clicked(self):
        self._train_action.setChecked(False)
        self._predict_action.setChecked(True)
        self._model = self._create_model(self._train_dict)

    #現在のサンプルデータからSVMモデルを生成する
    def _create_model(self, sample, param="-t 1"):
        if sample:
            labels = []
            data = []
            for i, (_, train) in enumerate(sorted(sample.items())):
                labels.extend([i] * len(train))
                data.extend(train)
            p = svm_parameter(param)
            prob = svm_problem(labels, data)
            return svm_train(prob, p)

    #現在のサンプルデータをモデルとして保存する
    @Slot()
    def _save_model(self):
        with open(self._model_file_edit.text(), 'w') as f:
            pickle.dump(self._train_dict, f)

    #ファイルからサンプルデータをモデルとして読み込む
    @Slot()
    def _load_model(self):
        with open(self._model_file_edit.text(), 'r') as f:
            self._train_dict = pickle.load(f)
            self._model = self._create_model(self._train_dict)
            self._train_combo.clear()
            for key in sorted(self._train_dict.keys()):
                self._train_combo.addItem(key)

    #モデルファイル選択ダイアログの呼び出し
    @Slot()
    def _select_file(self):
        filename = QFileDialog.getOpenFileName(self, QDir.currentPath())[0]
        self._model_file_edit.setText(filename)

    #bluetoothデバイスへの接続（テスト中）
    @Slot()
    def _connect_BT(self):
        self._BT_device = lb.selectdevice()
        if self._BT_device:
            device_address, device_name, _ = self._BT_device
            self._BT_state.setText(device_name)
            self._socket = lb.socket()
            self._socket.connect((device_address, 25))
            self._socket.send("connected")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    app.exec_()
