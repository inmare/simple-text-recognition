from text_generation import make_random_text
import image_processing as process
import image_cropping as crop
import image_data as data
import numpy as np
import time
import os
import debug
import graph
import yaml


def create_random_data():
    start_time = time.time()

    with open("./settings.yml", encoding="utf-8") as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    text_generation_info = settings["textGeneration"]

    seed = text_generation_info["seed"]
    text_h_len = text_generation_info["textHLen"]
    text_v_len = text_generation_info["textVLen"]
    page_len = text_generation_info["pageLen"]
    mode = text_generation_info["mode"]
    text_len = text_h_len * text_v_len * page_len

    random_text = make_random_text(seed, text_len, mode)

    print("Generating random text finished")
    debug.show_elapsed_time(start_time)

    image_info = settings["image"]
    image_path = image_info["imagePath"]
    image_perfix = image_info["imagePrefix"]
    start_page = image_info["startPage"]
    end_page = image_info["endPage"]

    image_file_paths = []
    for i in range(start_page, end_page + 1):
        image_file_path = image_path + image_perfix + str(i) + ".jpg"
        image_file_paths.append(image_file_path)

    textbox_info = settings["textbox"]
    denoise_info = settings["imageDenoise"]
    textbox_images = []

    for image_file_path in image_file_paths:
        textbox_image = crop.textbox_from_image(image_file_path, textbox_info, denoise_info)
        textbox_images.append(textbox_image)

    print("Extracting textbox finished")
    debug.show_elapsed_time(start_time)
    graph.show_image(textbox_images[0], (20, 20))

    line_footprint = np.ones((1, 25))
    line_images = []
    line_avg_height = 0
    page_cnt = 1
    line_cnt = 1

    for gray_image in gray_images:
        line_contours = data.get_line_contour_from_image(gray_image, line_footprint)

        for line_contour in line_contours:
            line_image = crop.line_from_textbox(gray_image, line_contour)
            line_avg_height += line_image.shape[0]
            line_images.append(line_image)
            # TODO debug 모듈에 오류감지함수 추가하기
            line_cnt += 1

        print(f"page {page_cnt} finished, line {line_cnt - 1} generated")
        page_cnt += 1
        line_cnt = 1

    line_avg_height = int(line_avg_height / len(line_images))

    print("Extracting line images finished")
    debug.show_elapsed_time(start_time)

    char_footprint = np.ones((1, 2))
    len_range = 5.
    line_gaps_width, line_x_coords = data.get_gaps_data(line_images, char_footprint)

    min_max_gaps, max_width = data.get_gaps_width(line_gaps_width, len_range)

    print("Extracting gap data finished")

    char_image_len = line_avg_height * (max_width + 1)
    char_data_array_len = char_image_len + max_width + line_avg_height + 1

    skeleton_thresh = 0.5
    char_size_dict = {
        "char_w": max_width,
        "char_h": line_avg_height,
        "data_len": char_data_array_len
    }

    line_cnt = 0
    char_cnt = 0

    char_data_array = np.zeros((len(line_images) * text_h_len, char_data_array_len))

    for i in range(len(line_images)):
        line_image = line_images[i]
        gaps_width = line_gaps_width[i]
        x_coords = line_x_coords[i]
        min_max_gap = min_max_gaps[i]

        line_start_time = time.time()

        for j in range(len(gaps_width)):
            gap_width = gaps_width[j]
            x_start, x_end = x_coords[j], x_coords[j + 1]
            y_start, y_end = 0, line_image.shape[0]

            if gap_width > min_max_gap[1]:
                chars_image = line_image[y_start:y_end, x_start:x_end]
                sep_chars = crop.char_from_chars(chars_image, skeleton_thresh, char_size_dict, min_max_gap)
                for k in range(sep_chars.shape[0]):
                    char_data_array[char_cnt:] = sep_chars[k, :]
                    char_cnt += 1
            else:
                char_image = line_image[y_start:y_end, x_start:x_end]
                char_image = process.move_char_image_to_center(char_image, skeleton_thresh, char_size_dict)
                char_data = data.add_stack_value_to_image(char_image)
                char_data_array[char_cnt, :] = char_data
                char_cnt += 1
        debug.show_elapsed_time_per_line(start_time, line_start_time, line_cnt + 1, char_cnt)
        line_cnt += 1

    npz_path = "./data/ascii-data"
    np.savez_compressed(npz_path, char_data_array=char_data_array, random_text=random_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = round((elapsed_time % 60) * 10) / 10

    print(f"image shape: {char_data_array.shape}\nelapsed time:{elapsed_min}m{elapsed_sec}s")


if __name__ == "__main__":
    create_random_data()
