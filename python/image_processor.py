import math

import cv2
import dlib
import numpy as np
from PIL import ImageEnhance, Image
from rembg import remove, new_session

# constant values
BRIGHTNESS_FACTOR = 1.2
INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 480
INPUT_PHOTO_PATH = '/usr/share/enrollment/images/photo_input.jpg'
OUTPUT_PHOTO_DPI = 400
OUTPUT_PHOTO_REMOVED_BG_PATH = "/usr/share/enrollment/images/removed_bg.png"  # for test purpose
OUTPUT_PHOTO_COMPRESSED_RESOLUTION = 64
OUTPUT_PHOTO_COMPRESSED_PATH = '/usr/share/enrollment/images/photo_compressed.png'
OUTPUT_PHOTO_RESOLUTION = 300
OUTPUT_PHOTO_PATH = '/usr/share/enrollment/images/photo.png'
PADDING = 0  # add padding if necessary
PREDICTOR_MODEL_PATH = '/usr/share/enrollment/model/model.dat'
SQUARE_RED_BOX_COR = (100, 38, 540, 478)  # (x1, y1, x2, y2) as per square redbox(400x400) size on UI
LINE_THICKNESS = 4
# Test which model gives best result and then set accordingly
"""
Different Model Names: 
            1. u2net
            2. isnet-general-use
"""

REMBG_MODEL_NAME = "u2net"  # OR REMBG_MODEL_NAME = "isnet-general-use"


def slope(x1, y1, x2, y2):
    return math.atan2((y2 - y1), (x2 - x1)) * (180 / math.pi)


def dx(x1, x2):
    return abs(x2 - x1)


def enhance_and_save_img(image: Image):
    brightness_enhancer = ImageEnhance.Brightness(image)
    brighten_img = brightness_enhancer.enhance(float(BRIGHTNESS_FACTOR))

    brighten_img.resize((OUTPUT_PHOTO_COMPRESSED_RESOLUTION, OUTPUT_PHOTO_COMPRESSED_RESOLUTION),
                        Image.LANCZOS).save(OUTPUT_PHOTO_COMPRESSED_PATH,
                                            "PNG",
                                            dpi=(OUTPUT_PHOTO_DPI, OUTPUT_PHOTO_DPI), optimize=False,
                                            quality=100)

    brighten_img.resize((OUTPUT_PHOTO_RESOLUTION, OUTPUT_PHOTO_RESOLUTION), Image.LANCZOS).save(
        OUTPUT_PHOTO_PATH, "PNG",
        dpi=(OUTPUT_PHOTO_DPI, OUTPUT_PHOTO_DPI), optimize=False,
        quality=100)
    print("Valid image")


def remove_bg_and_crop_img(image: Image) -> Image:
    session = new_session(model_name=REMBG_MODEL_NAME)
    # adjust alpha_matting_erode_size value to change edge blurring; alpha matting must be set to True
    bg_removed_img = remove(image, alpha_matting=True, session=session, post_process_mask=True)
    # bg_removed_img.save(OUTPUT_IMAGE_REMOVED_BG_PATH)  # Test purpose

    actual_x1, actual_y1, actual_x2, _ = bg_removed_img.getbbox()

    redbox_x1 = SQUARE_RED_BOX_COR[0]
    redbox_y1 = SQUARE_RED_BOX_COR[1]
    redbox_x2 = SQUARE_RED_BOX_COR[2]
    redbox_y2 = SQUARE_RED_BOX_COR[3]

    # trim transparent left, top and right sides (no extra transparent pixel in the bottom side) if there is padding
    # in the image redbox
    if actual_x1 > redbox_x1 and actual_x2 < redbox_x2 and actual_y1 > redbox_y1:
        while redbox_x1 - 2 < actual_x1 and redbox_x2 > actual_x2 + 2 and redbox_y1 - 4 < actual_y1:
            redbox_x1 += 2
            redbox_x2 -= 2
            redbox_y1 += 4

        if actual_x1 > redbox_x1 and actual_y1 > redbox_y1:
            while redbox_x1 - 2 < actual_x1 and redbox_y1 - 2 < actual_y1:
                redbox_x1 += 2
                redbox_y1 += 2

        if actual_x2 < redbox_x2 and actual_y1 > redbox_y1:
            while redbox_x2 > actual_x2 + 2 and redbox_y1 - 2 < actual_y1:
                redbox_x2 -= 2
                redbox_y1 += 2

        return bg_removed_img.crop((redbox_x1, redbox_y1, redbox_x2, redbox_y2))
    else:
        return bg_removed_img.crop(SQUARE_RED_BOX_COR)


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_MODEL_PATH)
    frame = cv2.imread(INPUT_PHOTO_PATH)
    detected_face = detector(frame, 0)
    face_count = len(detected_face)
    # make sure only one face is in the frame.
    if face_count == 0:
        ill = 0.2126 * frame[..., 2] + 0.7152 * frame[..., 1] + 0.722 * frame[..., 0]
        if ill.mean() > 110:
            print("Message= Adjust camera or face position")
        else:
            print("Message= Illumination not good enough")
    elif face_count > 1:
        print("Message= More than one person detected in frame")
    else:
        for _, d in enumerate(detected_face):
            x_min, y_min, x_max, y_max = (d.left(), d.top(), d.right(), d.bottom())
            x_min = x_min - 50
            x_max = x_max + 50
            y_max = y_max + 50
            y_min = max(0, y_min - abs(y_min - y_max) / 2.4)
            y_max = min(frame.shape[0], y_max + abs(y_min - y_max) / 9)
            x_min = max(0, x_min - abs(x_min - x_max) / 7)
            x_max = min(frame.shape[1], x_max + abs(x_min - x_max) / 7)
            x_max = min(x_max, frame.shape[1])
            # line thickness (negligible pixel)
            if x_min < (SQUARE_RED_BOX_COR[0] - LINE_THICKNESS) or y_min < (
                    SQUARE_RED_BOX_COR[1] - LINE_THICKNESS) or x_max > (SQUARE_RED_BOX_COR[2] + LINE_THICKNESS):
                print("Message= Move your body to fit in the box.")
                return

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
                                print("Message= Blurred image. Please come closer to the camera.")
                            else:
                                enhance_and_save_img(remove_bg_and_crop_img(Image.open(INPUT_PHOTO_PATH)))


main()
