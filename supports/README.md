Mount or copy a support bank here (`bank.json` + the image/mask files it references).
Each class has five example slots; `model.shots` uses the first 1 or 5. Classes without enough
valid examples are skipped when `classes: auto`.

    python -m triffid_uav bank ready --bank supports --shots 5
    python -m triffid_uav bank validate --bank supports --classes window,building
    python -m triffid_uav bank init --bank my_bank          # empty managed layout
    python -m triffid_uav bank add --bank my_bank --class window --slot 1 --image img.png --mask mask.png

Manifest (`triffid.uav.support-bank.v1`), paths relative to bank.json:

    {"schema_version": "triffid.uav.support-bank.v1",
     "classes": {"window": [{"id": "example_01", "image": "window/01/image.png", "mask": "window/01/mask.png"}, ...]}}

Masks are single-channel; foreground is any non-zero pixel, or `foreground_value` for indexed masks.
Support images are not committed (site imagery).
