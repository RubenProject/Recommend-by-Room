import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QInputDialog, QLineEdit, QFileDialog, QLabel
from PyQt5.QtCore import pyqtSlot, Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen
from complex_model import Complex_model



MODEL_LOC = '../nets/mixed_3_300_100_3.h5'


def parse_filename(filename):
    return filename.split('/')[-1:][0]


class App(QWidget):
    abr_full_map = {'Ac':'Action', 'Adu':'Adult', 'Adv':'Adventure', 'An':'Animation', 
            'B':'Biography', 'Co':'Comedy', 'Cr':'Crime', 'Dr':'Drama', 
            'Do':'Documentary', 'Fam':'Family', 'Fan':'Fantasy', 'Fi':'Film-Noir', 
            'Ga':'Game-Show', 'Hi':'History', 'Ho':'Horror', 'Mu1':'Music', 
            'Mu2':'Musical', 'My':'Mystery', 'N':'News', 'Re':'Reality-TV', 
            'Ro':'Romance', 'Sc':'Science-Fiction', 'Sh':'Short', 'Sp':'Sport', 
            'Ta':'Talk-Show', 'Th':'Thriller', 'Wa':'War', 'We':'Western',
            }
    top = []
    rec_movie = ''
#TODO: center image
#TODO: get recommended movie title
#TODO: increase fontsize
#TODO: center text
    def __init__(self, model_loc):
        super().__init__()
        self.model = Complex_model(model_loc)
        self.width = 1280
        self.height = 960
        self.topbar_width = 1280
        self.topbar_height = 100
        self.preview_width = 1280
        self.preview_height = 710
        self.botbar_width = 1280
        self.botbar_height = 150
        self.padding = 10

        self.setWindowTitle("Recommend by Room | DEMO")
        self.setGeometry(100, 100, self.width, self.height) 

        self.file_label = QLabel('No file selected', self)
        self.file_label.move(self.topbar_width / 2, self.padding)

        img_button = QPushButton('Open Image', self)
        img_button.move(self.topbar_width / 2, self.padding + 20)
        img_button.clicked.connect(self.open_file)

        run_button = QPushButton('Recommend me!', self)
        run_button.move(self.topbar_width / 2 + img_button.width(), self.padding + 20)
        run_button.clicked.connect(self.run_exp)

        self.img_preview = QLabel(self)
        self.img_preview.move(self.padding, self.topbar_height + self.padding)
        self.img_preview.resize(self.preview_width - self.padding * 2, 
                                self.preview_height - self.padding * 2)

        self.top_label = QLabel('Top matching genres:\n\n\n\n', self)
        self.top_label.move(self.botbar_width / 8, self.height - self.botbar_height + self.padding)

        self.rec_label = QLabel('Recommended movie:            \n', self)
        self.rec_label.move(self.botbar_width / 2, self.height - self.botbar_height + self.padding)

        self.show()


    def update(self):
        if self.filename:
            _filename = parse_filename(self.filename)
            self.file_label.setText(_filename)
            pixmap = QPixmap(self.filename)
            pixmap = pixmap.scaled(QSize(self.preview_width, self.preview_height), 
                                    aspectRatioMode=Qt.KeepAspectRatio)
            self.img_preview.setPixmap(pixmap)
        else:
            self.file_label.setText('No file selected')

        if len(self.top) == 3:
            self.top_label.setText('Top matching genres:'
                            + '\n1: ' + self.abr_full_map[self.top[0]]
                            + '\n2: ' + self.abr_full_map[self.top[1]] 
                            + '\n3: ' + self.abr_full_map[self.top[2]])
        else:
            self.top_label.setText('Top matching genres:\n\n\n\n')

        if self.rec_movie != '':
            self.rec_label.setText('Recommended movie:\n' + self.rec_movie)

        self.show()


    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(0, self.topbar_height, self.width, self.topbar_height)
        qp.drawLine(0, self.topbar_height + self.preview_height, self.width, 
                    self.topbar_height + self.preview_height)
        qp.end()




    @pyqtSlot()
    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.filename, _ =  QFileDialog.getOpenFileName(self, 'Open Image', '../tests', 
                "Images (*.png)", options=options)
        self.update()

    @pyqtSlot()
    def run_exp(self):
        if self.filename:
            self.top = self.model.run_test(self.filename)
            print(self.top)
        self.update()
        self.rec_movie = self.model.get_rec(self.top)
        print(self.rec_movie)
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App(MODEL_LOC)
    sys.exit(app.exec_())



