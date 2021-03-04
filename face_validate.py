# USAGE
# python face_validate.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import base64
from face_compare.images import get_face
from face_compare.model import facenet_model, img_to_encoding
from pydantic import BaseModel
from fastapi import FastAPI, Body

# construct the argument parse and parse the arguments


class Image64(BaseModel):
    data_face: str
    data_id: str


# if __name__ == '__main__':
#     ap = argparse.ArgumentParser(description='Face Comparison Tool')
#
#     # ap.add_argument('--image-one', dest='image_one', type=Path, required=True, help='Input Image One')
#     # ap.add_argument('--image-two', dest='image_two', type=Path, required=True, help='Input Image Two')
#     # ap.add_argument('-s', '--save-to', dest='save_dest', type=Path, help='Optionally save the cropped faces on disk. Input directory to save them to')
#     # args = ap.parse_args()
#
#     image1 = '/image.png'
#     image2 = '/image1.png'
#     run(image1, image2)

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
prototxt = 'deploy.prototxt.txt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'

app = FastAPI()
@app.post('/faceValidate')
async def faceValidate(req_body : Image64 = Body(...)):
    image = open('id_card_tmp.png', "wb")
    image.write(base64.b64decode(req_body.data_id))
    image.close()
    image = open('face.png', "wb")
    image.write(base64.b64decode(req_body.data_face))
    image.close()
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image_idCard = cv2.imread('id_card_tmp.png')
    (h, w) = image_idCard.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_idCard, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image_idCard, (startX, startY), (endX, endY), (255, 0, 0), 2)

            cv2.putText(image_idCard, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            image_idCard = image_idCard[startY:endY, startX:endX]
            image_face = cv2.imread('face.png')
            # image_tmp = open('id_card_tmp.png', "wb")
            # image_tmp.write(image_idCard)
            # image_tmp.close()
            #run(image_idCard, image_face, save_dest=None)
    # show the output image
    # y=2606
    # x=1175
    # h=3012
    # w=168
    cv2.imshow("Output", image)
    cv2.waitKey(0)

@app.post('/dataExtract')
async def dataExtract(req_body : Image64 = Body(...)):
    return 'test'


# def run(image_one, image_two, save_dest=None):
#     # Load images
#     # print(type(image_one))
#     # face_one = get_face(cv2.imread(str(image_one), 1))
#     # face_two = get_face(cv2.imread(str(image_two), 1))
#
#     # Optionally save cropped images
#     # if save_dest is not None:
#     #     print(f'Saving cropped images in {save_dest}.')
#     #     cv2.imwrite(str(save_dest.joinpath('face_one.png')), face_one)
#     #     cv2.imwrite(str(save_dest.joinpath('face_two.png')), face_two)
#
#     # load model
#     model = facenet_model(input_shape=(3, 96, 96))
#
#     # Calculate embedding vectors
#     embedding_one = img_to_encoding(image_one, model)
#     embedding_two = img_to_encoding(image_two, model)
#
#     dist = np.linalg.norm(embedding_one - embedding_two)
#     print(f'Distance between two images is {dist}')
#     if dist > 0.7:
#         print('These images are of two different people!')
#     else:
#         print('These images are of the same person!')
#
