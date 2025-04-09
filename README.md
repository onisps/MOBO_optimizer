# 🧠 Simulation-Guided Optimization of Parametric Leaflet Geometries Using Bayesian Design Space Exploration

A Python-based framework integrating finite element analysis (FEA) with multi-objective Bayesian optimization for the automated design and evaluation of cardiovascular leaflet structures.

---

## 📁 Project Structure

```
.
├── configuration/            # YAML config files for optimization/test runs
├── optimizer.py              # Core optimization runner using PhysBO
├── test.py                   # Testing framework (e.g. run predefined configs)
├── utils/                    # Modular utilities for geometry, FEA, results, and visuals
│   ├── abaqusWF/             # Abaqus-related workflows
│   └── *.py                  # Core functionality (geometry, FEA, plots, IO)
```

---

## 🚀 Key Features

- **Flexible Geometry Generation**:
  Parametrically model complex 3D leaflet geometries via `create_geometry_utils.py`.

- **Automated Abaqus Integration**:
  Run, monitor, and post-process FEA simulations for single or multi-leaflet models (`compute_utils.py`, `problem.py`).

- **Multi-Objective Optimization**:
  Use Gaussian process-based Bayesian optimization (via PhysBO) to minimize/maximize design metrics like:
  - Maximum von Mises stress (`VMS`)
  - Opening area fraction (`LMN_open`)
  - Coaptation quality (`LMN_close`)
  - Helicopter effect (deformation artifact)

- **Comprehensive Post-Processing**:
  Extract and visualize simulation metrics with `fea_results_utils.py` and `visualize.py`.

---

## ⚙️ Usage

### 1. Configure Optimization

Edit `configuration/config_leaf.yaml`:

```yaml
parameters:
  HGT: [8.0, 14.0]
  ANG: [10, 50]
  LAS: [5.0, 10.0]
  ...
objectives:
  - LMN_open
  - Smax
  - VMS
```

### 2. Run Optimization

```bash
python optimizer.py
```

This will:
- Parse parameter bounds and objectives
- Generate geometries
- Run Abaqus simulations in batch
- Evaluate performance
- Log and visualize optimization progress

### 3. Visualize Results

Use `utils/visualize.py` to generate convergence plots and Pareto fronts.

---

## 📊 Example Outputs

- **Convergence Plot**: Objective value progression over iterations
- **Pareto Front**: Trade-offs between objectives
- **3D Geometry**: Exported `.stl` meshes from parametric designs
- **Simulation Data**: Stress and displacement results from Abaqus `.odb` parsing

---

## 🧪 Dependencies

- Python ≥ 3.8
- Abaqus CAE (command-line interface required)
- [PhysBO](https://github.com/PreferredAI/physbo)
- `hydra`, `numpy`, `open3d`, `trimesh`, `matplotlib`, `seaborn`, `psutil`, `scipy`, `pandas`, `plotly`

Install via:

```bash
pip install -r requirements.txt
```

> ⚠️ **Note**: Abaqus must be installed and callable via command line.

---

## 🧬 Applications

This pipeline is adaptable to:
- Valve leaflet design optimization
- Soft robotics surface design
- Bio-inspired compliant mechanism studies
- Any simulation-driven inverse design framework

---

## 🧠 Authors & Credits

Developed by researchers focused on **simulation-based design of compliant structures**. Contact for academic collaborations or integration with new solvers.

---

## 📄 License

TBA
