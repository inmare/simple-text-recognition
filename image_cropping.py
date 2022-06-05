import image_processing as process
import image_data as data
from skimage import measure
import numpy as np
import graph


def edges_from_image(image: np.ndarray, top_margin: int, side_margin: int) -> np.ndarray:
    image_w = image.shape[1]
    image_h = image.shape[0]
    cropped_image = image[top_margin:image_h - side_margin, top_margin:image_w - side_margin]

    return cropped_image


def textbox_from_image(image_file_path, textbox_info):
    gray_image = process.make_gray_image(image_file_path)

    top_margin = textbox_info["topMargin"]
    side_margin = textbox_info["sideMargin"]
    center_image = edges_from_image(gray_image, top_margin, side_margin)

    clear_image = process.reduce_image_noise(center_image, textbox_info["noiseThresh"])
    cropped_image = textbox_from_image(clear_image, textbox_info["footprint"])

    dilated_image = process.make_dilated_image(image, footprint)

    contours = measure.find_contours(dilated_image, 0)

    textbox_contour = max(contours, key=len)
    min_x, min_y, max_x, max_y = data.get_coords_from_contour(textbox_contour)

    padding = footprint.shape[0] + 20
    cropped_image = image[min_y - padding:max_y + padding, min_x - padding:max_x + padding]

    return cropped_image


def line_from_textbox(textbox_image, line_contour):
    min_x, min_y, max_x, max_y = data.get_coords_from_contour(line_contour)
    line_contour_centered = data.fit_contour_to_image(line_contour, min_x, min_y)
    line_image = textbox_image[min_y:max_y + 1, min_x:max_x + 1]

    masked_line_image = process.make_masked_image_from_contour(line_image, line_contour_centered)

    return masked_line_image


def char_from_chars(chars_image, skeleton_thresh, char_size_dict, min_max_gap):
    data_len = char_size_dict["data_len"]

    image_data = np.zeros((1, data_len))

    # 정확한 결과를 위해 dilated된 이미지 대신 binary_image를 사용함
    binary_chars_image = process.make_binary_image(chars_image, thresh_correction=0.0)
    chars_image_vstack = data.get_image_stack_value(binary_chars_image, mode="v")
    crop_points = data.get_crop_point_from_vstack(chars_image_vstack, min_max_gap)

    for i in range(len(crop_points)):
        crop_x_start = crop_points[i]
        if i != len(crop_points) - 1:
            crop_x_end = crop_points[i + 1]
        else:
            crop_x_end = chars_image.shape[1]

        # 자르는 지점에 보정값으로 crop_x_end에 1을 추가함
        char_image = chars_image[:, crop_x_start:crop_x_end + 1]
        char_image = process.move_char_image_to_center(char_image, skeleton_thresh, char_size_dict)

        # vstack, hstack 값 추가
        char_data = data.add_stack_value_to_image(char_image)
        image_data = np.append(image_data, char_data[np.newaxis, :], axis=0)

    image_data = image_data[1:, :]
    return image_data


if __name__ == "__main__":
    pass
