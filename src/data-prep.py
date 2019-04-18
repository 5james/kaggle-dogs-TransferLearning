import os
from PIL import Image
import kaggle
import tarfile
import xml.etree.ElementTree as ET

DOWNLOAD_DIR = '../download/'
DATA_DIR = '../data/'

if __name__ == "__main__":
    if not os.path.isdir(DOWNLOAD_DIR):
        os.mkdir(DOWNLOAD_DIR)
    if not (os.path.isdir(DOWNLOAD_DIR + 'Annotation') and not os.path.isdir(DOWNLOAD_DIR + 'Images')):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('jessicali9530/stanford-dogs-dataset', path=DOWNLOAD_DIR, unzip=True)
        tar = tarfile.open(DOWNLOAD_DIR + 'images.tar', "r:")
        tar.extractall(path=DOWNLOAD_DIR)
        tar.close()
        tar = tarfile.open(DOWNLOAD_DIR + 'annotations.tar', "r:")
        tar.extractall(path=DOWNLOAD_DIR)
        tar.close()

    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    for breed in os.listdir(DOWNLOAD_DIR + 'Annotation/'):
        if not os.path.isdir(DATA_DIR + breed):
            os.mkdir(DATA_DIR + breed)
        for fileDog in os.listdir(DOWNLOAD_DIR + 'Annotation/' + breed):
            tree = ET.parse(DOWNLOAD_DIR + 'Annotation/' + breed + '/' + fileDog)
            bndboxXML = tree.getroot().findall('object')[0].find('bndbox')
            xmin = int(bndboxXML.find('xmin').text)
            xmax = int(bndboxXML.find('xmax').text)
            ymin = int(bndboxXML.find('ymin').text)
            ymax = int(bndboxXML.find('ymax').text)
            img = Image.open(DOWNLOAD_DIR + 'Images/' + breed + '/' + fileDog + '.jpg')
            img = img.crop((xmin, ymin, xmax, ymax))
            img = img.convert('RGB')
            img.save(DATA_DIR + breed + '/' + fileDog + '.jpg')
