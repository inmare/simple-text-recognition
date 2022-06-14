from skimage import io, color, filters, morphology, exposure, restoration, util
import scipy.ndimage as ndimage
import image_cropping as crop
import image_data as data
import numpy as np
import graph


def reduce_image_noise(image: np.ndarray, denoise_info: dict):
    """
    이미지의 노이즈를 줄여줌

    :param image: 노이즈를 줄이고 싶은 이미지
    :param denoise_info: denoise함수에 대한 정보를 담은 dict
    :returns: 노이즈가 줄어든 이미지를 반환함
    """
    denoise_amount = denoise_info["denoiseAmount"]
    image_tv = restoration.denoise_tv_chambolle(image, weight=denoise_amount)

    intensity_range = tuple(denoise_info["intensityRange"])
    percentiles = np.percentile(image_tv, intensity_range)
    image_minmax_scaled = exposure.rescale_intensity(image_tv, in_range=tuple(percentiles))

    # TODO np.subtract이 유의미한 결과를 가져오는지 확인 후 추가여부 결정
    # thresh = denoise_info["noiseThresh"]
    # image_noise_reduced = np.subtract(image_minmax_scaled, image_minmax_scaled, out=image_minmax_scaled,
    #                                   where=image_minmax_scaled < thresh)
    #
    # return image_noise_reduced

    return image_minmax_scaled


def make_gray_image(file_path: str) -> np.ndarray:
    """
    file_path로부터 이미지를 읽어 흑백이미지를 반환함

    :param file_path: 이미지 파일의 경로
    :returns: 흑백이미지를 반환함
    """
    image = io.imread(file_path)
    gray_image = color.rgb2gray(image)
    # 이미지의 값이 0과 1 사이가 아닌 0과 255 사이인 경우가 있어서 float형태로 한 번 더 변환
    gray_image = util.img_as_float(gray_image)

    return gray_image


def make_binary_image(gray_image, thresh_correction=-0.1, is_invert=True):
    """
    흑백이지미를 binary 이미지로 변환한 이미지를 반환함

    :param gray_image: dilation을 적용할 이미지
    :param thresh_correction: 1로 변환할 픽셀의 보정값. 값이 작을수록 binary정도가 강해짐
    :param is_invert: 이미지를 반전시킬 것인지. 기본값은 True
    :returns: binary 이미지를 반환함
    """
    thresh = filters.threshold_mean(gray_image)
    binary_image = gray_image > thresh + thresh_correction

    if is_invert:
        return np.invert(binary_image)
    else:
        return binary_image


def make_dilated_image(image: np.ndarray, footprint: np.ndarray, is_invert: bool = True,
                       thresh_correction: float = None) -> np.ndarray:
    """
    dilation이 적용된 binary 이미지를 반환함

    :param image: dilation을 적용할 이미지
    :param footprint: dilation을 적용하는 정도
    :param is_invert: 이미지를 반전시킬 것인지. 기본값은 True
    :param thresh_correction: binary 이미지를 만들 시 적용할 thresh correction. 기본값은 None
    :returns: dilation이 적용된 이미지를 반환함
    """
    if thresh_correction is not None:
        binary_image = make_binary_image(image, thresh_correction, is_invert)
    else:
        binary_image = make_binary_image(image, is_invert=is_invert)

    dilated_image = morphology.dilation(binary_image, footprint)

    return dilated_image


def make_masked_image_from_contour(image, contour):
    mask = np.zeros(image.shape, dtype=bool)
    mask[np.round_(contour[:, 0]).astype(np.int),
         np.round_(contour[:, 1]).astype(np.int)] = 1
    mask = ndimage.binary_fill_holes(mask)
    masked_image = mask * (1 - image)

    return 1 - masked_image


def move_char_image_to_center(char_image, skeleton_thresh, char_size_dict):
    image_h = char_size_dict["char_h"]
    image_w = char_size_dict["char_w"]

    blank_char_image = np.ones((image_h, image_w))
    # 오류를 방지하기 위해 기존의 이미지에 여백을 1픽셀을 준 새 이미지 생성
    padded_char_image = np.ones((char_image.shape[0] + 2, char_image.shape[1] + 2))
    padded_char_image[1:char_image.shape[0] + 1, 1:char_image.shape[1] + 1] = char_image

    # TODO np.subtract 사용 대신 새로운 함수 하나 만들기
    # inverted_char_image = np.subtract(1 - padded_char_image, 1 - padded_char_image,
    #                                   where=padded_char_image > 0.5)
    # TODO thresh_correction부분도 설정에 추가하기
    # binary_char_image = make_binary_image(inverted_char_image, thresh_correction=0., is_invert=False)
    binary_char_image = make_binary_image(padded_char_image)

    filled_char_image = ndimage.binary_fill_holes(binary_char_image)

    labeled_image, label_num = ndimage.label(1 - filled_char_image)
    masked_image = labeled_image == 0

    try:
        min_x, min_y, max_x, max_y = data.get_coords_from_binary_image(masked_image)

        # :, ;, ?...와 같이 요소가 2개인 글자들의 경우
        if label_num > 1:
            masked_image = labeled_image == 1
            mx, my, Mx, My = data.get_coords_from_binary_image(masked_image)
            min_x = mx if mx < min_x else min_x
            min_y = my if my < min_y else min_y
            max_x = Mx if Mx > max_x else max_x
            max_y = My if My > max_y else max_y

    except Exception as e:
        # graph.show_char(padded_char_image, binary_char_image)
        graph.show_image(binary_char_image, (1, 1))
        raise

    # 보정을 위해 max_x에 1을 더해줌. 추후에 수정 가능
    cropped_char_image = padded_char_image[min_y:max_y - 1, min_x:max_x + 1]
    char_w = max_x - min_x + 1
    char_h = max_y - min_y - 1

    x_start = int(round((image_w - char_w) / 2))
    y_start = int(round((image_h - char_h) / 2))

    x_end = x_start + char_w
    y_end = y_start + char_h

    blank_char_image[y_start:y_end, x_start:x_end] = cropped_char_image

    return 1 - blank_char_image


if __name__ == "__main__":
    pass
