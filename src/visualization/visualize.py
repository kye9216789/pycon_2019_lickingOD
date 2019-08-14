import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.quiz.quiz7 import bbox2delta


def draw_base_anchor(anchor_generator, line_size=1):
    base_anchors = anchor_generator.base_anchors
    assert base_anchors.size(1) == 4
    min_coord = base_anchors.min()
    base_anchors = base_anchors - min_coord + 1

    board_size = int(base_anchors.max()) + 5
    board = np.zeros((board_size, board_size, 3), dtype=np.uint8)

    for anchor in base_anchors:
        x1, y1, x2, y2 = [int(i) for i in anchor]
        board = cv2.rectangle(board, (x1, y1), (x2, y2), (255, 255, 255), line_size)

    plt.figure(figsize=(10, 10))
    plt.imshow(board)
