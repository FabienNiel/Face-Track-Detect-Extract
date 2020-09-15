"""
Blazeface image loader
"""

import logging
import numpy as np
import torch
import cv2
from skimage import io
from blazeface.blazeface import BlazeFace
from typing import Dict, Any, List

from constants import BLAZEFACE_WEIGHTS_PATH, BLAZEFACE_ANCHORS_PATH, RECTANGLE_POSITION_USER

logger = logging.getLogger(__name__)


class BlazeFaceLoader:
    def __init__(self, config_imageloader: Dict[str, Any]):
        self.H = config_imageloader['H']
        self.W = config_imageloader['W']
        self.C = config_imageloader['C']

        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = BlazeFace().to(self.gpu)

        self.net.load_weights(BLAZEFACE_WEIGHTS_PATH)
        self.net.load_anchors(BLAZEFACE_ANCHORS_PATH)

    def detect_faces(self, img: np.ndarray) -> List:
        """
        Returns a list of size n x 4 where n is the number of faces on img and the 4 integers are
        the ymin, xmin, ymax, xmax coordinates of each face
        :param img: Image captured by the terminal
        :type img: np.ndarray
        :return: list of the ymin, xmin, ymax, xmax coordinates of each face (size nx4 where n is
        the number of faces in img)
        :rtype: list
        """
        list_coordinates = []
        dim = (128, 128)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        detections = self.net.predict_on_image(resized)
        if len(detections) == 0:
            return None
        else:
            # a face was detected
            for detection in detections:
                print(detection)
                ymin = detection[0] * img.shape[0]
                xmin = detection[1] * img.shape[1]
                ymax = detection[2] * img.shape[0]
                xmax = detection[3] * img.shape[1]

                ymin -= 0.5 * ymin
                xmin -= 0.3 * xmin
                ymax += 0.1 * ymax
                xmax += 0.1 * xmax

                ymin = int(ymin)
                xmin = int(xmin)
                ymax = int(ymax)
                xmax = int(xmax)

                list_coordinates.append([ymin, xmin, ymax, xmax])

            return list_coordinates

    @staticmethod
    def crop_face(img: np.ndarray, coordinates: List):
        # Crop faces
        ymin, xmin, ymax, xmax = coordinates

        # check to be inside image size
        if ymin < 0:
            ymin = 0
        if ymax > img.shape[0]:
            ymax = img.shape[0]
        if xmin < 0:
            xmin = 0
        if xmax > img.shape[1]:
            xmax = img.shape[1]
        # check validity
        if ymax - ymin < 1 or xmax - xmin < 1:
            xmin = ymin = 0
            ymax, xmax = img.shape[:2]

        return img[ymin:ymax, xmin:xmax, :]

    def select_face(self, list_coordinates):
        # The face needs to be in the appropriate place (indicated on the screen).
        # There should be only one face in this area

        # TO DO : do once you know the position of the indication on the screen etc.
        # + write the return messages depending on each situation to indicate to the user what he should do
        if list_coordinates is not None:
            coordinates = list_coordinates[0]
            return coordinates
        else:
            return None

    @staticmethod
    def crop_img_rectangle(img):
        (xmin, ymin), (xmax, ymax) = RECTANGLE_POSITION_USER

        return img[ymin:ymax, xmin:xmax, :]

    def load(self, img_path: str):
        pass

    def load_from_path(self, img_path: str) -> List[np.ndarray]:
        """

        :param img_path:
        :type img_path:
        :return:
        :rtype:
        """
        img = io.imread(img_path)


        img_selected = self.load_from_img(img)

        return img_selected

    def load_from_img(self, img: np.ndarray) -> List[np.ndarray]:
        """

        :param img_path:
        :type img_path:
        :return:
        :rtype:
        """

        # Crop the image inside the rectangle of position of the user
        img_rect = BlazeFaceLoader.crop_img_rectangle(img)

        # Look for the faces inside this rectangle
        list_coordinates = self.detect_faces(img_rect)
        print(list_coordinates)

        if list_coordinates is None:
            logger.info('Please place yourself in the green rectangle')
            return None
        else:
            if len(list_coordinates) == 1:
                resized_img = cv2.resize(img_rect, (self.H, self.W), interpolation=cv2.INTER_CUBIC)

                return resized_img

            elif len(list_coordinates) > 1:
                logger.info('Please be alone in the green rectangle')
                return None



