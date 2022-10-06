import matplotlib.pyplot as plt
import torch


def handle_3d_image_tensor(data, ax):
    assert type(data) == torch.Tensor
    if str(data.device) != "cpu":
        data = data.detach().cpu()
    data = data - data.min()
    data = data / data.max()
    data_vis = data.permute(1, 2, 0)
    ax.imshow(data_vis)


def handle_3d_embedding_tensor(data, ax):
    assert type(data) == torch.Tensor
    if str(data.device) != "cpu":
        data = data.detach().cpu()
    data = data[:3]
    data = data - data.min()
    data = data / data.max()
    data_vis = data.permute(1, 2, 0)
    ax.imshow(data_vis)


TYPE_HANDLER = dict(
    image=handle_3d_embedding_tensor,
    embedding=handle_3d_embedding_tensor,
)


class GridVisualizer:
    def __init__(
        self,
        save_path=None,
        grid=(1,1),
    ):
        self.save_path = save_path
        self.row, self.column = grid
        self.fig, self.axs = plt.subplots(
            self.row,
            self.column,
            squeeze=False,
        )
        for i in range(self.row):
            for j in range(self.column):
                self.axs[i, j].axis('off')
                self.axs[i, j].margins(x=5, y=5, tight=True)
    
    def draw(self, data, type, row=1, column=1):
        if type not in TYPE_HANDLER.keys():
            raise ValueError(f"Type {type} is not in {TYPE_HANDLER.keys()}")
        ax = self.axs[row - 1, column - 1]
        TYPE_HANDLER[type](data, ax)
    
    def save(self):
        plt.tight_layout()
        plt.savefig(self.save_path, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()
    
    def show(self):
        self.fig.subplots_adjust(wspace=0.05)
        self.fig.tight_layout()
        plt.show()