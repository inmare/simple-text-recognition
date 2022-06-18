import image_processing as process
import image_data as data
from skimage import measure
import numpy as np
import graph


def edges_from_image(image: np.ndarray, top_margin: int, side_margin: int) -> np.ndarray:
    image_w = image.shape[1]
    image_h = image.shape[0]
    cropped_image = image[top_margin:image_h -
                          side_margin, top_margin:image_w - side_margin]

    return cropped_image


def textbox_from_image(image_file_path: str, textbox_info: dict, denoise_info: dict) -> np.ndarray:
    """
    이미지로부터 텍스트가 있는 구역을 추출함

    :param image_file_path: 이미지가 있는 파일 경로
    :param textbox_info: 텍스트 구역을 추출하기 위한 정보를 담은 dict
    :param denoise_info: denoise함수에 대한 정보를 담은 dict
    :returns: 텍스트 구역의 이미지를 ndarray형식으로 반환함
    """
    gray_image = process.make_gray_image(image_file_path)

    top_margin = textbox_info["topMargin"]
    side_margin = textbox_info["sideMargin"]
    center_image = edges_from_image(gray_image, top_margin, side_margin)

    clear_image = process.reduce_image_noise(center_image, denoise_info)

    textbox_footprint = np.ones(tuple(textbox_info["footprint"]))
    dilated_image = process.make_dilated_image(clear_image, textbox_footprint)

    contours = measure.find_contours(dilated_image, 0)
    textbox_contour = max(contours, key=len)
    min_x, min_y, max_x, max_y = data.get_coords_from_contour(textbox_contour)

    padding = textbox_footprint.shape[0] + 20
    cropped_image = clear_image[min_y - padding:max_y +
                                padding, min_x - padding:max_x + padding]

    return cropped_image


def line_from_textbox(textbox_image: np.ndarray, line_contour: np.ndarray) -> np.ndarray:
    """
    텍스트박스에서 줄이 있는 부분을 추출함

    :param textbox_image: numpy array형식의 텍스트박스 이미지
    :param line_contour: 줄의 contour의 위치를 담은 numpy array
    :returns: 줄의 이미지를 ndarray형식으로 반환함
    """
    min_x, min_y, max_x, max_y = data.get_coords_from_contour(line_contour)
    line_contour_centered = data.fit_contour_to_image(
        line_contour, min_x, min_y)
    line_image = textbox_image[min_y:max_y + 1, min_x:max_x + 1]

    masked_line_image = process.make_masked_image_from_contour(
        line_image, line_contour_centered)

    return masked_line_image


def char_from_chars(chars_image, chars_thresh, min_max_gap):
    seperated_chars = []

    # 정확한 결과를 위해 dilated된 이미지 대신 binary_image를 사용함
    binary_chars_image = process.make_binary_image(
        chars_image, thresh_correction=chars_thresh)
    tmp = binary_chars_image.astype(dtype=int)
    chars_image_vstack = data.get_image_stack_value(
        binary_chars_image, mode="v")
    crop_points = data.get_crop_point_from_vstack(
        chars_image_vstack, min_max_gap)

    for i in range(len(crop_points) - 1):
        crop_x_start = crop_points[i]
        crop_x_end = crop_points[i + 1]

        # 자르는 지점에 보정값으로 crop_x_end에 1을 추가함
        char_image = chars_image[:, crop_x_start:crop_x_end + 1]

        seperated_chars.append(char_image)

    return seperated_chars


if __name__ == "__main__":
    pass
