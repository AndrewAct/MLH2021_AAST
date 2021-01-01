import sys, os
#! pip install PyQt5

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

from PyQt5.QtCore import Qt

from PyQt5.QtGui import QPixmap

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter);
        self.setText('\n\n Drop Image Here \n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        '''
        )

    def setPixmap(self, image):
        super().setPixmap(image)


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)

        mainLayout = QVBoxLayout()
        self.photoViewr = ImageLabel()

        self.setLayout(mainLayout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].totalFile()
            self.set_image(file_path)

            event.accept()

        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewr.setPixmap(QPixmap(file_path))

app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())



