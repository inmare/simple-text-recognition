from text_generation import make_random_text
import image_processing as process
import image_cropping as crop
import image_data as data
import numpy as np
import time
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
    # graph.show_image(textbox_images[0], (20, 20))

    line_info = settings["line"]
    line_footprint = np.ones(tuple(line_info["footprint"]))
    line_images = []
    line_avg_height = 0
    page_cnt = 1
    line_cnt = 1

    for textbox_image in textbox_images:
        line_contours = data.get_line_contour_from_image(textbox_image, line_footprint)

        for line_contour in line_contours:
            line_image = crop.line_from_textbox(textbox_image, line_contour)
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

    char_info = settings["char"]
    # 현재는 Ubuntu Mono R을 쓰고 있기에 footprint의 값으로 (1,2)를 해둠
    # 조금 더 세로가 두꺼우면 get_gaps_data함수의 오류를 방지하는데 도움이 될 것 같아서임
    # 만약 Ubuntu Mono B를 사용한다면 상황에 따라 dilation대신 binarization 사용
    char_footprint = np.ones(tuple(char_info["footprint"]))
    outlier_range = settings["gapOutlierRange"]
    line_gaps_width, line_x_coords = data.get_gaps_data(line_images, char_footprint)

    min_max_gaps, max_width = data.get_gaps_width(line_gaps_width, outlier_range)

    print("Extracting gap data finished")
    debug.show_elapsed_time(start_time)

    char_image_len = line_avg_height * (max_width + 1)
    char_data_array_len = char_image_len + max_width + line_avg_height + 1

    skeleton_thresh = 0.5
    char_size_dict = {
        "char_w": max_width + 1,
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
            # 보정을 위해서 x_end에 1을 추가함
            x_start, x_end = x_coords[j], x_coords[j + 1]
            y_start, y_end = 0, line_image.shape[0]

            if gap_width > min_max_gap[1]:
                chars_image = line_image[y_start:y_end, x_start:x_end]
                # TODO 글자들을 분리하는 함수와 데이터를 추가하는 함수를 분리하기
                try:
                    sep_chars = crop.char_from_chars(chars_image, skeleton_thresh, char_size_dict, min_max_gap)
                except Exception as e:
                    print("chars")
                    print(e)
                    print(line_cnt, char_cnt)
                    b_imgs = process.make_binary_image(chars_image)
                    graph.show_char(chars_image, b_imgs)
                    continue
                for k in range(sep_chars.shape[0]):
                    char_data_array[char_cnt:] = sep_chars[k, :]
                    char_cnt += 1
            else:
                char_image = line_image[y_start:y_end, x_start:x_end]
                try:
                    char_image = process.move_char_image_to_center(char_image, skeleton_thresh, char_size_dict)
                except Exception as e:
                    print(e)
                    print(line_cnt, char_cnt)
                    b_img = process.make_binary_image(char_image, thresh_correction=-0.2)
                    graph.show_char(char_image, b_img)
                    continue
                    # return 1
                char_data = data.add_stack_value_to_image(char_image)
                char_data_array[char_cnt, :] = char_data
                char_cnt += 1
        debug.show_elapsed_time_per_line(start_time, line_start_time, line_cnt + 1, char_cnt)
        # TODO 글자들의 개수에 따라 이상을 감지하는 함수 추가하기
        # 정상적인 글자들의 데이터를 모은 뒤 어떤 글자들과도 일정 수준 일치하지 않으면 에러를 반환하는
        # 함수를 만들수도 있음
        line_cnt += 1

    npz_path = "./data/ascii-data"
    # np.savez_compressed(npz_path, train_data=char_data_array, label_data=random_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = round((elapsed_time % 60) * 10) / 10

    print(f"image shape: {char_data_array.shape}\nelapsed time:{elapsed_min}m{elapsed_sec}s")


if __name__ == "__main__":
    create_random_data()
