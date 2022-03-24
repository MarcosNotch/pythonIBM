class circulo(object):

    def __init__(self, radio, color):
        self.radio = radio
        self.color = color


    def agrandarRadio(self, r):
        self.radio = self.radio + r