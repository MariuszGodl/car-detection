"""Drawing boxes on images."""
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def show_image_with_boxes(img, boxes_xyxy, title="", color="lime", ax=None):
    """Draw pixel (x1, y1, x2, y2) boxes on ``img``; shows its own figure unless ``ax`` is given."""
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    for x1, y1, x2, y2 in boxes_xyxy:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1))
    ax.set_title(title)
    ax.axis("off")
    if own_fig:
        plt.show()
    return ax
