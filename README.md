# mosaiks-api

An example API for MOSAIKS.


## Setup

```
conda config --set channel_priority strict
conda env create --file environment.yml
conda activate mosaiks

# verify that the PyTorch can use the GPU
python -c "import torch; print(torch.cuda.is_available())"
```


## Running the server

```
conda activate mosaiks
python server.py --port 8080
```


## API

The `server.py` script exposes an HTTP server that responds to POST requests.

There is a single endpoint, `/featurizeSingle`, that expects input in the format:
```
{
    "latitude": latitude,
    "longitude": longitude
}
```

and returns the same JSON object with an additional `features` key that contains a 1024 length array that is the feature representation computed by RCF.

### Examples

See examples of how to query the API at `notebooks/Demo notebook.ipynb`.


## Data

The CSV files in `data/` were downloaded from the [MOSAIKS Code Ocean capsule](https://codeocean.com/capsule/6456296/tree/v2) from `data/int/applications/*/`.