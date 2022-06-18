import json
import time


def index_to_page(json_path: str, idx: int) -> list:
    """
    글자의 index를 [페이지, 줄, 글자]의 형식으로 만들어 줌

    :param json_path: 글자 정보를 담고 있는 json파일의 위치
    :param idx: 글자의 index
    :returns: [페이지, 글자, 줄] 형식의 리스트
    """
    # 글자 json데이터 불러오기
    with open(json_path) as info:
        char_info = json.load(info)

    text_h_len = char_info["textHLen"]
    text_v_len = char_info["textVLen"]
    page_text_len = text_h_len * text_v_len

    page_idx = 0
    if idx % page_text_len != 0:
        page_idx = idx // page_text_len + 1
    else:
        page_idx = idx // page_text_len

    idx_without_page = idx - ((page_idx - 1) * page_text_len)

    line_idx = 0
    if idx % text_h_len != 0:
        line_idx = idx_without_page // text_h_len + 1
    else:
        line_idx = idx_without_page - ((line_idx - 1) * text_h_len)

    char_idx = idx_without_page - ((line_idx - 1) * text_h_len)

    return [page_idx, line_idx, char_idx]


def show_elapsed_time_per_line(start_time, line_start_time, line_cnt, char_cnt, text_h_len):
    end_time = time.time()

    line_elapsed_time = end_time - start_time
    line_elapsed_min = int(line_elapsed_time // 60)
    line_elapsed_sec = round((line_elapsed_time % 60) * 10) / 10

    print(
        f"line {line_cnt} took: {line_elapsed_min}m{line_elapsed_sec}s expected:{line_cnt * text_h_len} actual:{char_cnt}")


def show_elapsed_time(start_time: float):
    """
    프로그램이 시작된 시간으로부터 걸린 시간을 보여줌

    :param start_time: 프로그램이 시작된 시간
    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = round((elapsed_time % 60) * 10) / 10

    print(f"elapsed time:{elapsed_min}m {elapsed_sec}s")


if __name__ == "__main__":
    pass
