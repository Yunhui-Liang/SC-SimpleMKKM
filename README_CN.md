# SC-SimpleMKKM（精简复现包）

本包保留 SC-SimpleMKKM 0.8.4.post2 主方法源码、MFeat 六视图小型真实数据、seed 0 配置和复现校验。未包含大体积传感器数据、合成数据、论文大规模输入及运行输出。

## 快速复现

需要 Python 3.10 或更新版本，推荐 Python 3.12。在仓库根目录依次运行：

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[ann-faiss,parallel]"
    python scripts/verify_data.py
    python -m scsimplemkkm.cli doctor
    python -m scsimplemkkm.cli plan --config configs/mfeat_rbf_self_tuned_seed0.json
    python -m scsimplemkkm.cli run-mv --config configs/mfeat_rbf_self_tuned_seed0.json
    python scripts/check_mfeat_reference.py

结果写入 outputs/。Linux/macOS 可用 source .venv/bin/activate 激活环境，其余 python 命令相同。

配置固定为 seed 0、每视图一个局部自适应 RBF 核、带宽近邻 local_scale_k=7、稀疏支持 support_k=10、SimpleMKKM 求解器和 batched Lloyd 离散化。MFeat 输入含 2,000 个样本、6 个视图、10 类。

历史参考值：ACC 0.872500、NMI 0.8794354695958252、ARI 0.8262795901539949。不同 NumPy、FAISS、BLAS 与 CPU 构建可能造成轻微差异；运行时间和内存是设备相关值。

MFeat 来源、CC BY 4.0 归属、处理说明及数据 SHA-256 见 data/README.md 和 data/SHA256SUMS。为缩小压缩包，未附原始 UCI 压缩档；已处理的六视图数据和标签已包含。
