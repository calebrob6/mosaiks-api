import rasterio
import rasterio.warp
import rasterio.mask
import shapely.geometry

import torch
from torch.utils.data import Dataset

class CustomNAIPDataset(Dataset):

    def __init__(self, points, fns, buffer=250):
        self.points = points
        self.fns = fns
        self.buffer = buffer

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):

        lon, lat = self.points[idx]
        fn = self.fns[idx]

        point_geom = shapely.geometry.mapping(
            shapely.geometry.Point(lon, lat)
        )

        with rasterio.Env():
            with rasterio.open(fn, "r") as f:
                point_geom = rasterio.warp.transform_geom("epsg:4326", f.crs.to_string(), point_geom)
                point_shape = shapely.geometry.shape(point_geom)
                mask_shape = point_shape.buffer(self.buffer).envelope
                mask_geom = shapely.geometry.mapping(mask_shape)
                out_image, _ = rasterio.mask.mask(f, [mask_geom], crop=True)

        out_image = (out_image[:3,:,:] / 255.0)
        out_image = torch.from_numpy(out_image).float()
        return out_image