import os
import argparse
import json

RASTERIO_BEST_PRACTICES = dict(
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE="200000000",
    GDAL_SWATH_SIZE="200000000",
    VSI_CURL_CACHE_SIZE="200000000",
)
os.environ.update(RASTERIO_BEST_PRACTICES)

import numpy as np
import torch
from torch.utils.data import DataLoader
import bottle

import rasterio
import rasterio.warp
import rasterio.mask
import shapely.geometry

from dataloaders import CustomNAIPDataset
from models import RCF, featurize
from NAIPTileIndex import NAIPTileIndex

DEVICE = torch.device("cpu")
INDEX = NAIPTileIndex(base_path="tmp/")
BUFFER_DISTANCE = 250  # in meters
NUM_FEATURES = 1024
MODEL = RCF(NUM_FEATURES).eval().to(DEVICE)

MAX_BATCH_SIZE = 1000

def enable_cors():
    """From https://gist.github.com/richard-flosi/3789163

    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    """
    bottle.response.headers["Access-Control-Allow-Origin"] = "*"
    bottle.response.headers[
        "Access-Control-Allow-Methods"
    ] = "PUT, GET, POST, DELETE, OPTIONS"
    bottle.response.headers[
        "Access-Control-Allow-Headers"
    ] = "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token"


def do_options():
    """This method is necessary for CORS to work (I think --Caleb)"""
    bottle.response.status = 204
    return


def featurize_naip_batched():
    bottle.response.content_type = "application/json"
    data = bottle.request.json

    if "latitudes" not in data:
        bottle.response.status = 500
        return json.dumps({
            "message": "'latitudes' is a required parameter but wasn't sent"}
        )
    if "longitudes" not in data:
        bottle.response.status = 500
        return json.dumps({
            "message": "'longitudes' is a required parameter but wasn't sent"}
        )
    if len(data["latitudes"]) != len(data["longitudes"]):
        bottle.response.status = 500
        return json.dumps({
            "message": "The 'latitudes' and 'longitudes' inputs are not the same length"}
        )
    if len(data["latitudes"]) > MAX_BATCH_SIZE:
        bottle.response.status = 500
        return json.dumps({
            "message": f"The maximum number of points you can process at once is {MAX_BATCH_SIZE}"}
        )


    points = []
    for i in range(len(data["latitudes"])):
        points.append(
            (data["longitudes"][i], data["latitudes"][i])
        )
    points = np.array(points)

    all_fns = []
    for i, (lon, lat) in enumerate(points):
        try:
            fns = list(set(INDEX.lookup_point(lat, lon)))
        except IndexError as e:
            bottle.response.status = 500
            return json.dumps({
                "message": str(e) + f" on point ({lon}, {lat})"}
            )
        fns = sorted(fns, key=lambda x: int(x.split("/")[6])) # sort by year
        fn = fns[-1]
        all_fns.append(fn)


    dataset = CustomNAIPDataset(points, all_fns, buffer=BUFFER_DISTANCE)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda x: x,
        pin_memory=False
    )

    x_all = np.zeros((points.shape[0], NUM_FEATURES), dtype=float)
    i = 0
    for images in dataloader: 
        for image in images:
            if image.shape[0] == 4:
                image = image.to(DEVICE)
                with torch.no_grad():
                    feats = MODEL(image.unsqueeze(0)).cpu().numpy()
                x_all[i] = feats
            elif image.shape[0] == 3:
                pass # this happens in at least Nevada for some points
            i += 1

    data["features"] = x_all.tolist()

    del dataset, dataloader

    bottle.response.status = 200
    return json.dumps(data)


def featurize_naip_single():
    bottle.response.content_type = "application/json"
    data = bottle.request.json

    if "latitude" not in data:
        bottle.response.status = 500
        return json.dumps({
            "message": "'latitude' is a required parameter but wasn't sent"}
        )
    if "longitude" not in data:
        bottle.response.status = 500
        return json.dumps({
            "message": "'longitude' is a required parameter but wasn't sent"}
        )

    lat, lon = None, None
    lat = data["latitude"]
    lon = data["longitude"]

    print(f"Featurizing ({lat}, {lon})")

    point_geom = shapely.geometry.mapping(
        shapely.geometry.Point(lon, lat)
    )

    try:
        fns = list(set(INDEX.lookup_point(lat, lon)))
    except IndexError as e:
        bottle.response.status = 500
        return json.dumps({
            "message": str(e)}
        )

    fns = sorted(fns, key=lambda x: int(x.split("/")[6])) # sort by year
    fn = fns[-1]

    with rasterio.open(fn, "r") as f:
        point_geom = rasterio.warp.transform_geom("epsg:4326", f.crs.to_string(), point_geom)
        point_shape = shapely.geometry.shape(point_geom)

        mask_shape = point_shape.buffer(BUFFER_DISTANCE).envelope
        mask_geom = shapely.geometry.mapping(mask_shape)

        out_image, _ = rasterio.mask.mask(f, [mask_geom], crop=True)

    features = featurize(out_image, MODEL, DEVICE)

    data["features"] = features.tolist()

    bottle.response.status = 200
    return json.dumps(data)


def main():
    parser = argparse.ArgumentParser(description="MOSAIKS Server")

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debugging",
        default=False,
    )
    parser.add_argument(
        "--host",
        action="store",
        dest="host",
        type=str,
        help="Host to bind to",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        action="store",
        dest="port",
        type=int,
        help="Port to listen on",
        default=4042,
    )

    args = parser.parse_args()

    # Setup the bottle server
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)

    app.route("/featurizeNAIPSingle", method="OPTIONS", callback=do_options)
    app.route("/featurizeNAIPSingle", method="POST", callback=featurize_naip_single)

    app.route("/featurizeNAIPBatched", method="OPTIONS", callback=do_options)
    app.route("/featurizeNAIPBatched", method="POST", callback=featurize_naip_batched)

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.verbose,
        "server": "tornado",
        "reloader": False,
    }
    app.run(**bottle_server_kwargs)


if __name__ == "__main__":
    main()
