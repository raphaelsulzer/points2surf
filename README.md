# Points2Surf for Point Clouds with Visibility Information

This repository contains the implementation of 
Shape As Points for Point Clouds with Visibility Information as
described in the paper 
[Deep Surface Reconstruction for Point Clouds with Visibility Information](https://arxiv.org/abs/2202.01810).

The code is largely based on the [original repository](https://github.com/ErlerPhilipp/points2surf).
Besides the support for point clouds with visibility information, we have changed the code to use `.yaml` 
config files for providing many arguments
previously provided via `argparser`.

# Data

The data used in this repository can be downloaded [here](https://github.com/raphaelsulzer/dsrv-data).


# Reconstruction

For reconstructing e.g. the ModelNet10 dataset run

`python full_eval.py configs/pointcloud/modelnet/config --gpu_idx 0`

where `config` should be replaced with
- `modelnet.yaml` for reconstruction from a point cloud (traditional Shape As Points)
- `modelnet_sen.yaml` for reconstruction from a point cloud augmented with sensor vectors
- `modelnet_aux_gl.yaml` for reconstruction from a point cloud augmented with sensor vectors and auxiliary points





## References

If you find the code or data in this repository useful, 
please consider citing

```bibtex
@misc{sulzer2022deep,
      title={Deep Surface Reconstruction from Point Clouds with Visibility Information}, 
      author={Raphael Sulzer and Loic Landrieu and Alexandre Boulch and Renaud Marlet and Bruno Vallet},
      year={2022},
      eprint={2202.01810},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@InProceedings{ErlerEtAl:Points2Surf:ECCV:2020,
  title   = {{Points2Surf}: Learning Implicit Surfaces from Point Clouds}, 
  author="Erler, Philipp
    and Guerrero, Paul
    and Ohrhallinger, Stefan
    and Mitra, Niloy J.
    and Wimmer, Michael",
  editor="Vedaldi, Andrea
    and Bischof, Horst
    and Brox, Thomas
    and Frahm, Jan-Michael",
  year    = {2020},
  booktitle="Computer Vision -- ECCV 2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="108--124",
  abstract="A key step in any scanning-based asset creation workflow is to convert unordered point clouds to a surface. Classical methods (e.g., Poisson reconstruction) start to degrade in the presence of noisy and partial scans. Hence, deep learning based methods have recently been proposed to produce complete surfaces, even from partial scans. However, such data-driven methods struggle to generalize to new shapes with large geometric and topological variations. We present Points2Surf, a novel patch-based learning framework that produces accurate surfaces directly from raw scans without normals. Learning a prior over a combination of detailed local patches and coarse global information improves generalization performance and reconstruction accuracy. Our extensive comparison on both synthetic and real data demonstrates a clear advantage of our method over state-of-the-art alternatives on previously unseen classes (on average, Points2Surf brings down reconstruction error by 30{\%} over SPR and by 270{\%}+ over deep learning based SotA methods) at the cost of longer computation times and a slight increase in small-scale topological noise in some cases. Our source code, pre-trained model, and dataset are available at: https://github.com/ErlerPhilipp/points2surf.",
  isbn="978-3-030-58558-7"
  doi = {10.1007/978-3-030-58558-7_7},
}
```
