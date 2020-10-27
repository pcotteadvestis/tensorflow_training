```python
from gcsfs import GCSFileSystem as gcs
fs = gcs(project="my_project")
apath = "gs://bucket_name/a/remote/path"
for item in fs.glob(f"{apath}/*"):
    print(item)
```

```python
from pathlib import Path
apath = Path("a") / "path" / "something"
for item in apath.glob("*"):
    print(item)
```

```python
from transparentpath import TransparentPath as Path
Path.set_global_fs(
    "gcs",
    project="my_project",
    bucket="bucket_name"
)
apath = Path("a") / "path" / "something"
for item in apath.glob("/*"):
    print(item)
```
