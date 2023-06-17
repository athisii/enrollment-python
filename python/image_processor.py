import math

import cv2
import dlib
import numpy as np
from PIL import ImageEnhance, Image
from rembg import remove

BRIGHTNESS_FACTOR = 1.4
INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 480
INPUT_IMAGE_PATH = '/usr/share/enrollment/images/input.jpg'
OUTPUT_IMAGE_DPI = 300
OUTPUT_IMAGE_CROPPED_PATH = "/usr/share/enrollment/images/out.png"  # for test purpose
OUTPUT_IMAGE_COMPRESS_SUB_RESOLUTION = 64
OUTPUT_IMAGE_COMPRESS_SUB_PATH = '/usr/share/enrollment/croppedimg/compressedsub.png'
OUTPUT_IMAGE_SUB_RESOLUTION = 300
OUTPUT_IMAGE_SUB_PATH = '/usr/share/enrollment/croppedimg/sub.png'
PADDING = 30
PREDICTOR_MODEL_PATH = '/usr/share/enrollment/model/model.dat'


def slope(x1, y1, x2, y2):
    return math.atan2((y2 - y1), (x2 - x1)) * (180 / math.pi)


def dx(x1, x2):
    return abs(x2 - x1)


def enhance_and_save_img(image: Image):
    if image is None:
        raise "Received a None object"
    brightness_enhancer = ImageEnhance.Brightness(image)
    brighten_img = brightness_enhancer.enhance(float(BRIGHTNESS_FACTOR))

    brighten_img.resize((OUTPUT_IMAGE_COMPRESS_SUB_RESOLUTION, OUTPUT_IMAGE_COMPRESS_SUB_RESOLUTION),
                        Image.LANCZOS).save(OUTPUT_IMAGE_COMPRESS_SUB_PATH,
                                            "PNG",
                                            dpi=(OUTPUT_IMAGE_DPI, OUTPUT_IMAGE_DPI), optimize=False,
                                            quality=100)

    brighten_img.resize((OUTPUT_IMAGE_SUB_RESOLUTION, OUTPUT_IMAGE_SUB_RESOLUTION), Image.LANCZOS).save(
        OUTPUT_IMAGE_SUB_PATH, "PNG",
        dpi=(OUTPUT_IMAGE_DPI, OUTPUT_IMAGE_DPI), optimize=False,
        quality=100)
    print("Valid image")


def remove_bg_and_crop_img(image: Image) -> Image:
    if image is None:
        raise "Received a None object"
    bg_removed_img = remove(image, post_process_mask=True)
    # Convert the image to binary using threshold
    binary_image = bg_removed_img.convert('L').point(lambda p: p > 0 and 255)  # True and number --> number
    x1, y1, x2, y2 = binary_image.getbbox()
    x_diff = x2 - x1
    y_diff = y2 - y1
    if x_diff > OUTPUT_IMAGE_DPI:
        extra_pixel = (x_diff - OUTPUT_IMAGE_DPI) / 2
        x1 += extra_pixel
        x2 -= extra_pixel

    if y_diff > OUTPUT_IMAGE_DPI:
        y2 -= y_diff - OUTPUT_IMAGE_DPI

    return bg_removed_img.crop((x1 - PADDING, y1, x2 + PADDING, y2 + PADDING))


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_MODEL_PATH)

    frame = cv2.imread(INPUT_IMAGE_PATH)
    detected_face = detector(frame, 0)
    face_count = len(detected_face)
    if face_count == 0:
        ill = 0.2126 * frame[..., 2] + 0.7152 * frame[..., 1] + 0.722 * frame[..., 0]
        if ill.mean() > 110:
            print("Message= Adjust camera or face position")
        else:
            print("Message= Illumination not good enough")
    elif face_count > 1:
        print("Message= More than one person detected in frame")
    else:
        for k, d in enumerate(detected_face):
            # [(543, 281)(758, 496)]
            x_min, y_min, x_max, y_max = (d.left(), d.top(), d.right(), d.bottom())
            x_min = x_min - 50
            x_max = x_max + 50
            y_max = y_max + 50
            y_min = max(0, y_min - abs(y_min - y_max) / 2.4)
            y_max = min(frame.shape[0], y_max + abs(y_min - y_max) / 9)
            x_min = max(0, x_min - abs(x_min - x_max) / 7)
            x_max = min(frame.shape[1], x_max + abs(x_min - x_max) / 7)
            x_max = min(x_max, frame.shape[1])

            if int(x_min) <= 0 or int(x_max) >= frame.shape[1] or int(y_min) <= 0 or int(y_max) >= frame.shape[0]:
                print("Message= Face/Chest Going out of frame.....come to middle")
            else:
                shape = predictor(frame, d)
                np.matrix([[p.x, p.y] for p in shape.parts()])

                if abs(dx(shape.parts()[4].x, shape.parts()[3].x) - dx(shape.parts()[4].x, shape.parts()[1].x)) > 9:
                    if abs(slope(shape.parts()[2].x, shape.parts()[2].y, shape.parts()[0].x, shape.parts()[0].y)) > 3:
                        if slope(shape.parts()[2].x, shape.parts()[2].y, shape.parts()[0].x, shape.parts()[0].y) < 0:
                            print("Message= ROTATE Face CLOCK")
                        else:
                            print("Message= ROTATE Face ANTI CLOCK")
                    else:
                        if dx(shape.parts()[4].x, shape.parts()[3].x) - dx(shape.parts()[4].x, shape.parts()[1].x) < 0:
                            print('Message= ROTATE Face RIGHT')
                        else:
                            print('Message= ROTATE Face LEFT')
                else:
                    if abs(slope(shape.parts()[2].x, shape.parts()[2].y, shape.parts()[0].x, shape.parts()[0].y)) > 3:
                        if slope(shape.parts()[2].x, shape.parts()[2].y, shape.parts()[0].x, shape.parts()[0].y) < 0:
                            print("Message= ROTATE Face CLOCK")
                        else:
                            print("Message= ROTATE Face ANTI CLOCK")

                    else:
                        if abs(slope(shape.parts()[2].x, shape.parts()[2].y, shape.parts()[3].x,
                                     shape.parts()[3].y)) > 4 or abs(
                            slope(shape.parts()[1].x, shape.parts()[1].y, shape.parts()[0].x, shape.parts()[0].y)) > 4:
                            if (slope(shape.parts()[2].x, shape.parts()[2].y, shape.parts()[3].x,
                                      shape.parts()[3].y) < 0 < slope(shape.parts()[1].x, shape.parts()[1].y,
                                                                      shape.parts()[0].x, shape.parts()[0].y)):
                                print("Message= CHIN DOWN.... AND KEEP EYES NORMAL")
                            else:
                                print("Message= CHIN UP....AND KEEP EYES NORMAL")

                        else:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                            if fm < 60:
                                print("Message= Blurred Image")
                            else:
                                cropped_image = remove_bg_and_crop_img(Image.open(INPUT_IMAGE_PATH))
                                # cropped_image.save(OUTPUT_IMAGE_CROPPED_PATH) # for test purpose
                                enhance_and_save_img(cropped_image)


main()