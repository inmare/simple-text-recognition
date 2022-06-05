import numpy as np


class seed_random:
    def __init__(self, seed):
        self.seed = seed

    def random(self):
        self.seed = (1013904223 + 1664525 * self.seed) % 4294976296
        return self.seed / 4294967296

    def choice(self, array):
        idx = round(self.random() * len(array) - 0.5)
        return array[idx]


def make_random_text(seed: int, text_len: int, mode: str) -> np.ndarray:
    """
    주어진 길이만큼의 랜덤한 텍스트를 생성함

    :param seed: 랜덤 시드
    :param text_len: 생성할 텍스트의 길이
    :param mode: ascii와 hex 2개를 인자로 받음
    :returns: 랜덤한 텍스트의 아스키코드들을 ndarray형태로 반환함
    """
    random = seed_random(seed)
    char_list = []

    if mode == "ascii":
        char_list = [i for i in range(0x21, 0x7f)]
    elif mode == "hex":
        for i in range(0x21, 0x7f):
            if chr(i) in "0123456789abcdef":
                char_list.append(i)

    random.random()  # 더 랜덤한 값을 위해 랜덤 함수를 한 번 실행해 줌
    random_text = []

    for i in range(text_len):
        random_text.append(random.choice(char_list))

    return np.asarray(random_text, dtype=np.int8)


if __name__ == "__main__":
    pass
