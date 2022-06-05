from skimage import io, color, filters, morphology, exposure, restoration, util
import scipy.ndimage as ndimage
import image_cropping as crop
import image_data as data
import numpy as np
import graph


def reduce_image_noise(image: np.ndarray, thresh: float) -> np.ndarray:
    """
    이미지의 노이즈를 줄여줌

    :param image: 노이즈를 줄이고 싶은 이미지
    :param thresh: 디노이징 함수를 얼마나 적용할 것인가. 값이 높을수록 적용되는 정도가 커짐
    :returns: 노이즈가 줄어든 이미지를 반환함
    """
    # TODO 아래의 값들 중 수정이 필요한 값들은 json파일에 저장해두기
    image_tv = restoration.denoise_tv_chambolle(image, weight=0.04)
    percentiles = np.percentile(image_tv, (20, 60))
    image_minmax_scaled = exposure.rescale_intensity(image_tv, in_range=tuple(percentiles))
    # TODO np.subtract이 유의미한 결과를 가져오는지 확인 후 추가여부 결정
    image_noise_reduced = np.subtract(image_minmax_scaled, image_minmax_scaled, where=image_minmax_scaled < thresh)

    return image_noise_reduced


def make_gray_image(file_path: str) -> np.ndarray:
    """
    file_path로부터 이미지를 읽어 흑백이미지를 반환함

    :param file_path: 이미지 파일의 경로
    :returns: 흑백이미지를 ndarray형태로 반환함
    """
    image = io.imread(file_path)
    gray_image = color.rgb2gray(image)
    # 이미지의 값이 0과 1 사이가 아닌 0과 255 사이인 경우가 있어서 float형태로 한 번 더 변환
    gray_image = util.img_as_float(gray_image)

    return gray_image


def make_binary_image(gray_image, thresh_correction=-0.1, is_invert=True):
    thresh = filters.threshold_mean(gray_image)
    binary_image = gray_image > thresh + thresh_correction

    if is_invert:
        return np.invert(binary_image)
    else:
        return binary_image


def make_dilated_image(image, footprint):
    inverted_binary_image = make_binary_image(image)
    dilated_image = morphology.dilation(inverted_binary_image, footprint)

    return dilated_image


def make_masked_image_from_contour(image, contour):
    mask = np.zeros(image, dtype=bool)
    mask[np.round_(contour[:, 0]).astype(np.int),
         np.round_(contour[:, 1]).astype(np.int)] = 1
    mask = ndimage.binary_fill_holes(mask)
    masked_image = mask * (1 - image)

    return 1 - masked_image


def move_char_image_to_center(char_image, skeleton_thresh, char_size_dict):
    image_h = char_size_dict["avg_height"]
    image_w = char_size_dict["max_width"]

    blank_char_image = np.ones((image_h, image_w))
    # 오류를 방지하기 위해 기존의 이미지에 여백을 1픽셀을 준 새 이미지 생성
    padded_char_image = np.ones((char_image.shape[0] + 2, char_image.shape[1] + 2))
    padded_char_image[1:char_image.shape[0] + 1, 1:char_image.shape[1] + 1] = char_image

    # TODO np.subtract 사용 대신 새로운 함수 하나 만들기
    inverted_char_image = np.subtract(1 - padded_char_image, 1 - padded_char_image,
                                      where=1 - padded_char_image < skeleton_thresh)
    binary_char_image = make_binary_image(inverted_char_image, thresh_correction=0.1, is_invert=False)
    filled_char_image = ndimage.binary_fill_holes(binary_char_image)

    labeled_image, label_num = ndimage.label(1 - filled_char_image)
    masked_image = labeled_image == 0
    min_x, min_y, max_x, max_y = data.get_coords_from_binary_image(masked_image)

    # :, ;, ?...와 같이 요소가 2개인 글자들의 경우
    if label_num > 1:
        masked_image = labeled_image == 1
        mx, my, Mx, My = data.get_coords_from_binary_image(masked_image)
        min_x = mx if mx < min_x else min_x
        min_y = my if my < min_y else min_y
        max_x = Mx if Mx > max_x else max_x
        max_y = My if My > max_y else max_y

    # 보정을 위해 max_x에 1을 더해줌. 추후에 수정 가능
    cropped_char_image = padded_char_image[min_y:max_y, min_x:max_x + 1]
    char_w = max_x - min_x + 1
    char_h = max_y - min_y

    x_start = int(round((image_w - char_w) / 2))
    y_start = int(round((image_h - char_h) / 2))

    x_end = x_start + char_w
    y_end = y_start + char_h

    blank_char_image[y_start:y_end, x_start:x_end] = cropped_char_image

    return 1 - blank_char_image


if __name__ == "__main__":
    pass
