# VisionBoard data contract

VisionBoard joins images and text files by scene stem. For `images/A.png`, every optional
layer is read from `<configured-folder>/A.txt`.

## Built-in V7 source

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

The V7 Lowres source uses displayed images2x imagery. Deployed predictions are normalized,
`labels_simple` is already in images2x pixels, and `labels_detail_id` uses native pixels with
`instance_id_scale=2.0`.

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
    pred_scale=1.0,
    gt_scale=1.0,
    instance_id_scale=2.0,
    writable=False,
)
```

Runtime registrations are stored in `data/sources.json`, which is intentionally untracked.

