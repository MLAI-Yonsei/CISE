# QETB data

QETB data is supplied externally. Obtain it through [JARVIS-Tools](https://jarvis-tools.readthedocs.io/en/master/databases.html) or the [publisher dataset page](https://figshare.com/articles/dataset/15127788).

```bash
python -m pip install jarvis-tools
```

```python
from jarvis.db.figshare import data

rows = data("qe_tb")
```

Record the dataset version, checksum and splits used for each experiment. Follow the provider's license and citation requirements.

Downloading the dataset does not prepare the trained property models or calibration assets. Supply those separately as described in [ASSETS.md](ASSETS.md), and check that the available labels support the intended properties and calculation protocol.
