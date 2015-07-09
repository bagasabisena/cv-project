__author__ = 'bagas'

from PIL import Image
import align

pil_im = Image.open('../data/yalefaces/subject01.gif')

eyes = align.eye_detector(pil_im)
print eyes
