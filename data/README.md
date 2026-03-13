# Data

The dataset used in this research is not included in this repository.

You can download the complete dataset from the following permanent identifier:

🔗 **DOI:** [10.5281/zenodo.19001042](https://doi.org/10.5281/zenodo.19001042)

## Instructions
1.  Download the data from the link above.
2.  Extract the contents (if compressed).
3.  Place the extracted files and folders directly into this `data/` directory.

After downloading, place the contents in `data/cloud/` following this structure:

```bash
data/
├── cloud/
│   ├── mushroom/
│   │   ├── sample1/
│   │   │   ├── a.csv # x,y,z and intensity at wavelength 1320 nm
│   │   │   ├── b.csv # x,y,z and intensity at wavelength 1450 nm
│   │   │   └── c.csv # x,y,z and intensity at moisture index (MI)
│   │   ├── sample2/
│   │   └── ...
│   └── broccoli/
│       ├── sample1/
│       └── ...
└── mc.csv # ground-truth moisture per sample & region
```
