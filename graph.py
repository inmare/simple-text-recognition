from matplotlib import pyplot as plt
import numpy as np


def show_image(image, graph_size: tuple, title: str = None):
    """
    이미지를 그래프로 보여줌

    :param image: 보여주려는 이미지
    :param graph_size: (x, y)형태의 그래프 크기
    :param title: 그래프의 제목. 기본으로는 아무 제목도 설정되어 있지 않음
    """
    plt.figure(figsize=graph_size)
    plt.imshow(image)
    plt.axis("off")
    plt.margins(x=0, y=0)
    graph_title = title
    if graph_title:
        plt.title(graph_title)
    plt.show()


def show_char(char_img, binary_img):
    plt.figure(figsize=(2, 1))

    plt.subplot(121)
    plt.imshow(char_img)
    plt.axis("off")
    plt.margins(x=0, y=0)

    plt.subplot(122)
    plt.imshow(binary_img)
    plt.axis("off")
    plt.margins(x=0, y=0)

    plt.show()


if __name__ == "__main__":
    pass
