import image_processing as process
from skimage import measure, io
import numpy as np
import graph
import math


def get_coords_from_contour(contour) -> tuple:
    """
    contour의 좌표값들중 x와 y좌표의 최대, 최솟값을 반환함

    :param contour: ndarray형태의 x와 y좌표값들의 배열
    :returns: (min_x, min_y, max_x, max_y)의 형태로 좌표값을 반환함
    """
    y_coords, x_coords = np.hsplit(contour, 2)
    y_coords = np.hstack(y_coords)
    x_coords = np.hstack(x_coords)

    min_y = math.floor(np.min(y_coords))
    min_x = math.floor(np.min(x_coords))
    max_y = math.ceil(np.max(y_coords))
    max_x = math.ceil(np.max(x_coords))

    return min_x, min_y, max_x, max_y


def get_coords_from_binary_image(binary_image):
    x, = np.where(binary_image.sum(axis=0) >= 1)
    y, = np.where(binary_image.sum(axis=1) >= 1)

    # 보정을 위한 -1과 +2를 추가함
    min_x, max_x = x.min().astype(np.int8) - 1, x.max().astype(np.int8) + 2
    min_y, max_y = y.min().astype(np.int8) - 1, y.max().astype(np.int8) + 2

    # 보정에 의해 MinMax x, y값이 음수나 이미지 크기를 넘을 경우를 위해 값을 조정함
    binary_image_w = binary_image.shape[1] - 1
    binary_image_h = binary_image.shape[0] - 1

    min_x = 0 if min_x < 0 else min_x
    min_y = 0 if min_y < 0 else min_y
    max_x = binary_image_w if max_x > binary_image_w - 1 else max_x
    max_y = binary_image_h if max_y > binary_image_h - 1 else max_y

    return min_x, min_y, max_x, max_y


def get_line_contour_from_image(image: np.ndarray, line_footprint: np.ndarray, line_thresh: float) -> np.ndarray:
    """
    이미지로부터 줄들의 contour좌표를 가지고 있는 배열을 반환함

    :param image: 줄을 추출하고 싶은 이미지
    :param line_footprint: 줄을 추출하기 위한 dilation함수에서 인자로 넣을 footprint
    :param line_thresh: dilation함수에서 인자로 넣을 보정값
    :returns: 각 줄의 contour좌표가 있는 numpy배열
    """
    dilated_image = process.make_dilated_image(
        image, line_footprint, thresh_correction=line_thresh)
    contours = measure.find_contours(dilated_image, 0)
    # 이미지의 가로길이보다 contour의 길이가 더 길면 줄의 contour이라고 가정
    # 추후에 더 좋은 방법이 나오면 수정할 수도 있음
    line_contours = np.array([
        contour for contour in contours if len(contour) > image.shape[1]
    ], dtype=object)

    return line_contours


def fit_contour_to_image(contour, min_x, min_y):
    contour_y, contour_x = np.hsplit(contour, 2)
    contour_x = contour_x - min_x
    contour_y = contour_y - min_y
    centered_contour = np.append(contour_y, contour_x, axis=1)

    return centered_contour


def get_image_stack_value(image, mode):
    image_stack_value = None
    if mode == "v":
        image_h = image.shape[0]
        image_stack_value = np.sum(image, axis=0)
        image_stack_value = np.true_divide(image_stack_value, image_h)
    elif mode == "h":
        image_w = image.shape[1]
        image_stack_value = np.sum(image, axis=1)
        image_stack_value = np.true_divide(image_stack_value, image_w)

    return image_stack_value


def get_x_coords_from_vstack(image_vstack_value):
    image_w = image_vstack_value.shape[0]
    zero_x_coords = np.setdiff1d(np.arange(image_w), np.where(
        np.ceil(image_vstack_value * 10) != 0.))

    x_coords = []
    coords_cache = []

    for i in range(len(zero_x_coords)):
        coord = zero_x_coords[i]
        # 배열의 끝에 다다랐을 경우 coords_cache의 제일 앞 값을 x_coords에 저장
        if i == len(zero_x_coords) - 1:
            x_coords.append(coords_cache[0])
            coords_cache = []
            break

        if coord + 1 == zero_x_coords[i + 1]:
            coords_cache.append(coord)
        else:
            if 0 in coords_cache:
                x_coords.append(coord)
                coords_cache = []
            else:
                coords_cache.append(coord)
                coords_average = int(np.round_(np.average(coords_cache)))
                x_coords.append(coords_average)
                coords_cache = []

    return np.asarray(x_coords)


def reject_outliers(gaps_data, outlier_range):
    len_from_median = np.abs(gaps_data - np.median(gaps_data))

    return gaps_data[len_from_median < outlier_range]


def get_gaps_data(line_images, char_footprint):
    gaps_data = []
    line_gaps_width = []
    line_x_coords = []

    for line_image in line_images:
        dilated_line_image = process.make_dilated_image(
            line_image, char_footprint)
        line_vstack_value = get_image_stack_value(dilated_line_image, mode="v")

        x_coords = get_x_coords_from_vstack(line_vstack_value)
        gaps_width = np.diff(x_coords)

        line_gaps_width.append(gaps_width)
        line_x_coords.append(x_coords)

    return line_gaps_width, line_x_coords


def get_gaps_width(line_gaps_width, outlier_range):
    # 줄 별 min, max gap 저장
    min_max_gaps = []
    max_width = 0

    for gaps_width in line_gaps_width:
        gaps_data_no_outlier = reject_outliers(gaps_width, outlier_range)

        min_gap_width = gaps_data_no_outlier.min().astype(np.int8)
        max_gap_width = gaps_data_no_outlier.max().astype(np.int8)

        min_max_gaps.append((min_gap_width, max_gap_width))

        if max_width < max_gap_width:
            max_width = max_gap_width

    return min_max_gaps, max_width


def get_crop_point_from_vstack(image_vstack, min_max_gap):
    min_gap_width = min_max_gap[0]
    max_gap_width = min_max_gap[1]

    start_x = 0
    x_coords = []

    # start_point = np.where(np.floor(image_vstack[:min_gap_width] * 10) == 0.)
    # TODO min_gap_width 대신 더 좋은 기준을 추가할 수도 있음
    start_point = np.where(image_vstack[:min_gap_width] == 0.)

    if len(start_point[0]):
        start_x = np.argmax(start_point)

    x_coords.append(start_x)

    while start_x + max_gap_width < image_vstack.shape[0]:
        array_check_section = image_vstack[start_x +
                                           min_gap_width:start_x + max_gap_width]
        crop_point = np.argmin(array_check_section)
        crop_point += start_x + min_gap_width
        x_coords.append(crop_point)
        start_x = crop_point

    if image_vstack.shape[0] - start_x > min_gap_width:
        x_coords.append(image_vstack.shape[0] - 1)

    return x_coords


def add_stack_value_to_image(char_image):
    flatten_char_image = char_image.ravel()
    char_data = np.append(flatten_char_image,
                          get_image_stack_value(char_image, "v"))
    char_data = np.append(char_data, get_image_stack_value(char_image, "h"))

    return char_data


if __name__ == "__main__":
    pass
