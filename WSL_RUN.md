# Setup TensorFlow GPU di WSL2 untuk Menjalankan Notebook (.ipynb)

Panduan ini digunakan untuk setup environment Deep Learning menggunakan:
- WSL2 Ubuntu
- TensorFlow GPU
- NVIDIA CUDA
- VS Code Remote WSL
- Jupyter Notebook / `.ipynb`

Cocok untuk:
- CNN
- RNN/LSTM
- Image Segmentation
- Deep Learning Training dengan GPU NVIDIA

---

# 1. Requirement

## Hardware
- NVIDIA GPU (contoh: RTX 4060)

## Software
Install:
- WSL2
- Ubuntu (disarankan Ubuntu 22.04)
- NVIDIA Driver terbaru
- VS Code
- VS Code Extension:
  - Remote - WSL
  - Python
  - Jupyter

---

# 2. Cek GPU di Windows

Di CMD / PowerShell:

```bash
nvidia-smi
```

Kalau berhasil akan muncul:
- nama GPU
- CUDA Version
- VRAM

Contoh:

```text
NVIDIA GeForce RTX 4060
CUDA Version: 13.0
```

---

# 3. Cek GPU di WSL

Masuk WSL:

```bash
wsl
```

Lalu cek:

```bash
nvidia-smi
```

Kalau GPU muncul lagi berarti GPU passthrough WSL berhasil.

---

# 4. Install Python 3.10 di WSL

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip -y
```

---

# 5. Clone Repository

```bash
git clone <REPOSITORY_URL>
cd <NAMA_REPOSITORY>
```

Contoh:

```bash
git clone https://github.com/user/project.git
cd project
```

---

# 6. Buat Virtual Environment

```bash
python3.10 -m venv .venv
```

Aktifkan:

```bash
source .venv/bin/activate
```

Kalau aktif akan muncul:

```text
(.venv)
```

---

# 7. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

---

# 8. Install TensorFlow GPU

Install TensorFlow + CUDA dependency:

```bash
pip install tensorflow[and-cuda]
```

---

# 9. Install Jupyter Kernel

```bash
pip install notebook jupyter ipykernel
```

Register kernel:

```bash
python -m ipykernel install --user --name=tf-gpu
```

---

# 10. Setup LD_LIBRARY_PATH

Tambahkan CUDA library path:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
```

Agar permanen, masukkan ke:

```bash
~/.bashrc
```

Lalu apply:

```bash
source ~/.bashrc
```

---

# 11. Test TensorFlow GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Kalau berhasil:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

# 12. Buka Project di VS Code

Dari terminal WSL:

```bash
code .
```

Pastikan VS Code menunjukkan:

```text
WSL: Ubuntu
```

di kiri bawah.

---

# 13. Menjalankan Notebook (.ipynb)

## Buka notebook

Klik file `.ipynb`.

---

## Pilih kernel

Klik:

```text
Select Kernel
```

Pilih:

```text
Python 3.10 (.venv)
```

atau:

```text
tf-gpu
```

Pastikan menggunakan interpreter:

```text
/home/<USERNAME>/<PROJECT>/.venv/bin/python
```

---

# 14. Cek GPU di Notebook

Jalankan:

```python
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
```

---

# 15. Monitor GPU Usage

Di terminal lain:

```bash
watch -n 1 nvidia-smi
```

Saat training berjalan:
- GPU utilization naik
- VRAM usage naik
- Process Python muncul

---

# Notes

## Penting
Jangan install:
- `nvidia-driver` Linux manual
- CUDA toolkit manual dari apt
- cuDNN manual

Karena WSL2 sudah menggunakan driver NVIDIA dari Windows.

---

## Workflow Recommended

### VS Code
- coding
- debugging
- notebook

### WSL2
- TensorFlow GPU
- Linux environment

### GitHub
- version control

---

# Test Training Sederhana

```python
import tensorflow as tf
import numpy as np

x = np.random.rand(10000, 1024).astype(np.float32)
y = np.random.rand(10000, 10).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=10, batch_size=256)
```