import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from rasterio.windows import from_bounds
import rasterio as rio
from rasterio import features
from shapely.geometry import LineString, Polygon
import geopandas as gpd
import numpy as np
import pandas as pd
from skimage.transform import resize


class FSODataset(Dataset):
    def __init__(self, samples, full_images, full_masks, output_size):
        self.samples = samples
        self.full_images = full_images
        self.full_masks = full_masks
        self.output_size = output_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self.full_images[s["region"]]
        mask = self.full_masks[s["region"]]


        H, W = self.output_size, self.output_size
        h_img, w_img = img.shape[1:]
        row_start = max(0, int(s["row_start"]))
        row_end = min(h_img, int(s["row_end"]))
        col_start = max(0, int(s["col_start"]))
        col_end = min(w_img, int(s["col_end"]))

        # evitar crops vacíos
        if row_end <= row_start or col_end <= col_start:
            # devolver un dummy vacío o saltar el sample
            crop_img = np.zeros((img.shape[0], H, W), dtype=np.float32)
            crop_mask = np.zeros((H, W), dtype=np.int64)
        else:
            crop_img = img[:, row_start:row_end, col_start:col_end]
            crop_mask = mask[row_start:row_end, col_start:col_end]

            crop_img = resize(
                crop_img,
                (img.shape[0], H, W),
                order=1,
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.float32)

            crop_mask = resize(
                crop_mask,
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.int64)

        return torch.from_numpy(crop_img), torch.from_numpy(crop_mask).long().unsqueeze(0)

class FSO(pl.LightningDataModule):
    def __init__(self, root, batch_size=32, num_workers=4, output_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root=root
        self.output_size=output_size

        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.set_names = None

    def setup(self, stage=None):
        l1cbands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
        l2abands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

        allregions = [
            "accra_20181031",
            "biscay_20180419",
            "danang_20181005",
            "kentpointfarm_20180710",
            "kolkata_20201115",
            "lagos_20190101",
            "lagos_20200505",
            "london_20180611",
            "longxuyen_20181102",
            "mandaluyong_20180314",
            "neworleans_20200202",
            "panama_20190425",
            "portalfredSouthAfrica_20180601",
            "riodejaneiro_20180504",
            "sandiego_20180804",
            "sanfrancisco_20190219",
            "shengsi_20190615",
            "suez_20200403",
            "tangshan_20180130",
            "toledo_20191221",
            "tungchungChina_20190922",
            "tunisia_20180715",
            "turkmenistan_20181030",
            "venice_20180630",
            "venice_20180928",
            "vungtau_20180423"
        ]

        self.samples = []
        self.full_images = {}
        self.full_masks = {}

        for region in allregions:
            shapefile = os.path.join(self.root, "shapefiles", region + ".shp")
            imagefile = os.path.join(self.root, "scenes", region + ".tif")
            imagefilel2a = os.path.join(self.root, "scenes", region + "_l2a.tif")

            if os.path.exists(imagefilel2a):
                imagefile = imagefilel2a



            with rio.open(imagefile) as src:
                full_img = src.read()
                transform = src.transform
                crs = src.crs

            # keep only 12 bands: delete 10th band (nb: 9 because start idx=0)
            if (full_img.shape[0] == 13):  # is L1C Sentinel 2 data
                full_img = full_img[[l1cbands.index(b) for b in l2abands]]

            lines = gpd.read_file(shapefile).to_crs(crs)
            is_closed_line = lines.geometry.apply(line_is_closed)
            polys = lines.loc[is_closed_line].geometry.apply(Polygon)
            rasterize_geometries = pd.concat([lines.geometry, polys])

            full_mask = features.rasterize(
                rasterize_geometries,
                out_shape=(full_img.shape[1], full_img.shape[2]),
                transform=transform,
                all_touched=True
            )

            self.full_images[region] = full_img.astype("float32")
            self.full_masks[region] = full_mask.astype("int64")

            for geom in lines.geometry:
                left, bottom, right, top = geom.bounds
                r0, c0 = rio.transform.rowcol(transform, left, top)
                r1, c1 = rio.transform.rowcol(transform, right, bottom)

                self.samples.append(dict(
                    region=region,
                    row_start=min(r0, r1),
                    row_end=max(r0, r1),
                    col_start=min(c0, c1),
                    col_end=max(c0, c1)
                ))

        n_train = int(0.8 * len(self.samples))
        self.train_dataset = FSODataset(self.samples[:n_train], self.full_images, self.full_masks, self.output_size)
        self.val_dataset = FSODataset(self.samples[n_train:], self.full_images, self.full_masks, self.output_size)

    def within_image(self, geometry):
        left, bottom, right, top = geometry.bounds
        ileft, ibottom, iright, itop = self.imagebounds
        return ileft < left and iright > right and itop > top and ibottom < bottom

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)




def split_line_gdf_into_segments(lines):
    def segments(curve):
        return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))

    line_segments = []
    for geometry in lines.geometry:
        line_segments += segments(geometry)
    return gpd.GeoDataFrame(geometry=line_segments)

def line_is_closed(linestringgeometry):
    coordinates = np.stack(linestringgeometry.xy).T
    first_point = coordinates[0]
    last_point = coordinates[-1]
    return bool((first_point == last_point).all())
