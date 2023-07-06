import math

import cv2
import dlib
import numpy as np
from PIL import ImageEnhance, Image
from rembg import remove, new_session

# constant values
BRIGHTNESS_FACTOR = 1.2
INPUT_IMAGE_WIDTH = 640  # Camera max width
INPUT_IMAGE_HEIGHT = 480  # Camera max height
INPUT_IMAGE_PATH = '/usr/share/enrollment/images/input.jpg'
OUTPUT_IMAGE_DPI = 400
OUTPUT_IMAGE_REMOVED_BG_PATH = "/usr/share/enrollment/images/removed_bg.png"  # for test purpose
OUTPUT_IMAGE_COMPRESS_SUB_RESOLUTION = 64
OUTPUT_IMAGE_COMPRESS_SUB_PATH = '/usr/share/enrollment/croppedimg/compressed.png'
OUTPUT_IMAGE_SUB_RESOLUTION = 300  # need to be finalized
OUTPUT_IMAGE_SUB_PATH = '/usr/share/enrollment/croppedimg/sub.png'
PADDING = 0  # add padding if necessary
PREDICTOR_MODEL_PATH = '/usr/share/enrollment/model/model.dat'
PARENT_DIR = '/usr/share/enrollment/images'
MIN_HEIGHT = 400  # close-up camera for clarity
MAX_HEIGHT = INPUT_IMAGE_HEIGHT - 30
MIN_WIDTH = 400  # close-up camera for clarity
MAX_WIDTH = INPUT_IMAGE_WIDTH - 90

# Test which model gives best result and then set accordingly
"""
Different Model Names: 
            1. u2net
            2. isnet-general-use
"""

REMBG_MODEL_NAME = "u2net"  # OR REMBG_MODEL_NAME = "isnet-general-use"
face_coordinates = None


def slope(x1, y1, x2, y2):
    return math.atan2((y2 - y1), (x2 - x1)) * (180 / math.pi)


def dx(x1, x2):
    return abs(x2 - x1)


def enhance_and_save_img(image: Image):
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


def remove_bg_test(image: Image):
    session = new_session(model_name=REMBG_MODEL_NAME)
    # adjust alpha_matting_erode_size value to change edge blurring; alpha matting must be set to True
    bg_removed_img = remove(image, alpha_matting=True, session=session, post_process_mask=True)
    # bg_removed_img.save(OUTPUT_IMAGE_REMOVED_BG_PATH)  # Test purpose
    x1, y1, x2, y2 = bg_removed_img.getbbox()
    height = y2 - y1
    width = x2 - x1
    print(f'Height: {height}')
    print(f'Width: {width}')
    if MIN_HEIGHT <= height <= MAX_HEIGHT and MIN_WIDTH <= width <= MAX_WIDTH:
        enhance_and_save_img(bg_removed_img.crop((x1 - PADDING, y1, x2 + PADDING, y2 + PADDING)))
    else:
        print("Message= Fit your body in the red box.")


def remove_bg_and_crop_img(image: Image) -> Image:
    session = new_session(model_name=REMBG_MODEL_NAME)
    # adjust alpha_matting_erode_size value to change edge blurring; alpha matting must be set to True
    bg_removed_img = remove(image, alpha_matting=True, session=session, post_process_mask=True)
    # bg_removed_img.save(OUTPUT_IMAGE_REMOVED_BG_PATH)  # Test purpose
    x1, y1, x2, y2 = bg_removed_img.getbbox()
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
            x_min, y_min, x_max, y_max = (d.left(), d.top(), d.right(), d.bottom())
            global face_coordinates
            face_coordinates = d
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
                                # remove_bg_and_crop_img(Image.open(INPUT_IMAGE_PATH))
                                remove_bg_test(Image.open(INPUT_IMAGE_PATH))


main()
