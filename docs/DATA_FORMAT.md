# VisionBoard data contract

VisionBoard joins images and text files by scene stem. For `images/A.png`, every optional
layer is read from `<configured-folder>/A.txt`.

## Built-in V8 source

Set `VISIONBOARD_DATA_ROOT` to a directory containing:

```text
images/
predictions/stage1/
predictions/stage2/
predictions/stage2_raw/
predictions/stage2_vectors/
classes/stage1.txt
classes/stage2.txt
gt/labels_simple/
gt/labels_delete/
gt/labels_detail_id/
heatmaps/stage0/
explainability/gradcam/manifests/
explainability/gradcam/cams/
```

Only `images/` is required to discover scenes. Missing optional layers are reported through
the API and omitted from the corresponding UI feature.

## Coordinate spaces

Every overlay is converted to displayed-image pixels:

```text
displayed prediction = stored prediction × pred_scale
displayed GT         = stored GT × gt_scale
displayed details    = stored detailed geometry × instance_id_scale
```

The V8 Lowres source uses displayed images2x imagery. Deployed predictions are normalized,
`labels_simple` is already in images2x pixels, and `labels_detail_id` uses native pixels with
`instance_id_scale=2.0`.

## Stage0 heatmaps

`heatmaps/stage0/<scene>.npy` is a 2-D float array on the native raster grid. Values are
clipped to `[0,1]` by the API before Turbo colorization. The browser scales this grid to the
displayed images2x dimensions and applies the user-selected opacity.

## Grad-CAM contract

Each scene has `explainability/gradcam/manifests/<scene>.json`:

```json
{
  "version": 1,
  "predictions": {
    "stage2:7": {
      "file": "cams/stage2/<scene>__stage2_7.png",
      "bounds": [x1, y1, x2, y2],
      "models": ["YOLO8m"],
      "contributors": [{"model": "YOLO8m", "score": 0.81}],
      "method": "single highest-confidence ensemble contributor CAM"
    }
  }
}
```

Prediction IDs are stable file-line IDs: `stage1:<zero-based-line>` and
`stage2:<zero-based-line>`. Bounds are absolute displayed-image pixels. CAM files are
full-context color PNGs spatially aligned inside those bounds. File paths are relative to
the configured Grad-CAM root and cannot escape that root.

## Prediction formats

Deployed normalized Stage 1/2:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 confidence
```

Raw Stage 2:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name confidence
```

Stage 2 vector:

```text
x1 y1 x2 y2 x3 y3 x4 y4 winning_class winning_score
arty camo logistic missile radar smallvehicle tank vehicle
```

The last line is the semantic order of the eight numeric score fields, not literal text in
the file.

## Ground truth

`labels_simple` is DOTA-style and is the only evaluation truth:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

Delete regions are variable-length polygons ending in `Delete`. A prediction is ignored when
its centroid is inside a delete polygon.

Detailed-ID files contain 26 fields. Fields 1–25 are the detailed label record; field 26 is
the stable five-character instance ID. The backend joins detailed records to simple records
by polygon IoU, because `labels_simple` may intentionally remap partly camouflaged objects.

## Registering another source

Add a `SOURCES` entry in `app/config.py`, or write a runtime registration through
`POST /api/sources`:

```python
dict(
    name="example",
    title="Example dataset",
    images="/data/example/images",
    image_glob="*.png",
    strip_suffix="",
    stage1="/data/example/predictions/stage1",
    stage2="/data/example/predictions/stage2",
    prediction_format="lowres_normalized",
    stage1_classes="/data/example/classes/stage1.txt",
    stage2_classes="/data/example/classes/stage2.txt",
    gt="/data/example/gt/labels_simple",
    delete="/data/example/gt/labels_delete",
    instance_ids="/data/example/gt/labels_detail_id",
    stage0="/data/example/heatmaps/stage0",
    gradcam="/data/example/explainability/gradcam",
    pred_scale=1.0,
    gt_scale=1.0,
    instance_id_scale=2.0,
    writable=False,
)
```

Runtime registrations are stored in `data/sources.json`, which is intentionally untracked.

