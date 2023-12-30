import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader
import torch.nn.functional  as F

from core.ae import AE
from core.dataset import DefectsDataset
from core.transforms import get_test_transforms

torch.manual_seed(42)
pl.seed_everything(42)

def threshold_selector():
    model = AE.load_from_checkpoint("models/model.ckpt")
    model.eval()
    mse_losses = list()
    intermediate_dataset = DefectsDataset(
        "data/external/defects/proliv", transform=get_test_transforms(32)
    )
    intermediate_dataloader = DataLoader(intermediate_dataset, batch_size=1)
    with torch.no_grad():
        for image in intermediate_dataloader:
            reconstacted_image = model(image.to("cuda"))
            mse_losses.append(F.mse_loss(image, reconstacted_image.cpu()).item())
    print(sorted(mse_losses))

    #подбираем bandwidth
    # Используйте GridSearchCV для подбора оптимального bandwidth
    param_grid = {'bandwidth': np.linspace(0.0001, 1.0, 10)}
    kde = KernelDensity(kernel='gaussian')
    grid_search = GridSearchCV(kde, param_grid, cv=5)
    grid_search.fit(np.array(mse_losses).reshape(-1, 1))

    # Получите оптимальное значение bandwidth
    optimal_bandwidth = grid_search.best_params_['bandwidth']

    kde_spills = KernelDensity(kernel='gaussian', bandwidth=optimal_bandwidth).fit(np.array(mse_losses).reshape(-1, 1))

    # Generate a range of values for MSE loss
    range_values = np.linspace(min(mse_losses), max(mse_losses), 1000).reshape(-1, 1)

    # Compute the log density for each value
    log_density_spills = kde_spills.score_samples(range_values)

    # Find the minimum of the log density as the threshold
    threshold = range_values[np.argmin(log_density_spills)][0]
    print(threshold)


if __name__ == "__main__":
    threshold_selector()
