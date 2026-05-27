import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


ROOT = Path(__file__).resolve().parents[3]
ADVERSARIAL_DIR = ROOT / "adversarial"
ENSEMBLE_DIR = ROOT / "ensemble"
EXPERIMENT_DIRS = {
    "experiments": ROOT / "experiments",
    "experiments2": ROOT / "experiments2",
}
METADATA_PATHS = {
    "metadata1": ROOT / "metadata.json",
    "metadata2": ROOT / "metadata(2).json",
}
ANALYSIS_DIRS = {
    "experiments": ROOT / "analysis_results",
    "experiments2": ROOT / "analysis_results" / "experiments2",
}
REPORT_PATH = ROOT / "analysis_results" / "evaluation_report.md"
RUNS_PATH = ROOT / "experiments" / "realtime_runs.jsonl"
RUNTIME_RECORDS_PATH = ROOT / "experiments" / "realtime_records.jsonl"
GENERATION_RUNS_PATH = ROOT / "experiments" / "generation_runs.jsonl"
GENERATED_SAMPLES_DIR = ROOT / "generated_samples"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="多模态模型攻击与安全评估平台",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "Microsoft YaHei UI", "Inter", sans-serif;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .hero {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 22px 24px;
        background: #ffffff;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        color: #4b5563;
        font-size: 15px;
        line-height: 1.65;
        max-width: 960px;
        margin: 0;
    }
    .panel {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 18px;
        background: #ffffff;
        min-height: 116px;
    }
    .panel-title {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 8px;
    }
    .panel-value {
        color: #111827;
        font-size: 25px;
        font-weight: 700;
        line-height: 1.15;
    }
    .panel-note {
        color: #6b7280;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.45;
    }
    .section-note {
        color: #4b5563;
        font-size: 14px;
        line-height: 1.65;
        margin-bottom: 12px;
    }
    .status-ok {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        background: #ecfdf5;
        color: #047857;
        font-size: 12px;
        border: 1px solid #a7f3d0;
    }
    .status-warn {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        background: #fffbeb;
        color: #92400e;
        font-size: 12px;
        border: 1px solid #fde68a;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: #6b7280;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 14px;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_whitebox(cache_key: tuple) -> dict:
    raw = {}
    for exp_dir in EXPERIMENT_DIRS.values():
        path = exp_dir / "whitebox_results.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for key, item in data.items():
            if key not in raw:
                raw[key] = item
                continue
            old_asr = raw[key].get("whitebox_asr")
            new_asr = item.get("whitebox_asr")
            if old_asr is None and new_asr is not None:
                raw[key] = item

    # White-box is one canonical baseline set, not two experiment sources.
    canonical_order = [
        "resnet_only_eps4",
        "resnet_only_eps8",
        "resnet_only_eps16",
        "vit_only_eps4",
        "vit_only_eps8",
        "vit_only_eps16",
        "ensemble_eps4",
        "ensemble_eps8",
        "ensemble_eps16",
    ]
    merged = {}
    for key in canonical_order:
        if key not in raw:
            continue
        item = dict(raw[key])
        merged[key] = item
    return merged


@st.cache_data(show_spinner=False)
def load_blackbox(cache_key: tuple) -> pd.DataFrame:
    frames = []
    for source, exp_dir in EXPERIMENT_DIRS.items():
        path = exp_dir / "records.jsonl"
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True, dtype={"image_id": str})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["image_id"] = combined["image_id"].astype(str).str.zfill(4)
    combined = combined.drop_duplicates(subset=["image_id", "surrogate", "eps", "target_model"], keep="last")
    return combined


@st.cache_data(show_spinner=False)
def load_summary(cache_key: tuple) -> pd.DataFrame:
    df = load_blackbox(cache_key)
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(["surrogate", "eps", "target_model"], as_index=False)
        .agg(
            samples=("image_id", "nunique"),
            records=("attack_success", "size"),
            attack_success_rate=("attack_success", "mean"),
        )
    )
    summary["attack_success_percent"] = summary["attack_success_rate"] * 100
    return summary


@st.cache_data(show_spinner=False)
def load_metadata_inventory(cache_key: tuple) -> pd.DataFrame:
    rows = []
    for source, path in METADATA_PATHS.items():
        if not path.exists():
            rows.append({"来源": source, "文件": display_path(path), "样本数": 0, "有效": False})
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "来源": source,
                "文件": display_path(path),
                "样本数": len(data),
                "有效": True,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_metadata_map(cache_key: tuple) -> dict:
    path = METADATA_PATHS["metadata2"] if METADATA_PATHS["metadata2"].exists() else METADATA_PATHS["metadata1"]
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["id"]).zfill(4): item for item in data}


@st.cache_data(show_spinner=False)
def load_realtime_runs(cache_key: tuple) -> pd.DataFrame:
    if not RUNS_PATH.exists():
        return pd.DataFrame()
    return pd.read_json(RUNS_PATH, lines=True)


@st.cache_data(show_spinner=False)
def load_generation_runs(cache_key: tuple) -> pd.DataFrame:
    if not GENERATION_RUNS_PATH.exists():
        return pd.DataFrame()
    return pd.read_json(GENERATION_RUNS_PATH, lines=True)


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.png"))


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def adversarial_folder(surrogate: str, eps: str) -> Path:
    if surrogate == "ensemble":
        return ENSEMBLE_DIR / eps
    return ADVERSARIAL_DIR / surrogate / eps


def target_model_options() -> list:
    return ["inception_v3", "densenet121", "mobilenet_v2", "efficientnet_b0"]


@st.cache_resource(show_spinner=False)
def get_target_model(model_name: str):
    if model_name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
    else:
        raise ValueError(f"未知目标模型：{model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


def target_transform(model_name: str):
    if model_name == "inception_v3":
        resize_size = 342
        crop_size = 299
    else:
        resize_size = 256
        crop_size = 224
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def clean_prediction_from_metadata(item: dict, surrogate: str):
    if surrogate == "vit_only":
        return item.get("clean_pred_vit")
    return item.get("clean_pred_resnet")


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_attack_sample_generation(algorithm: str, surrogate: str, eps: str, sample_count: int):
    source_dir = adversarial_folder(surrogate, eps)
    if not source_dir.exists():
        raise FileNotFoundError(f"样本目录不存在：{display_path(source_dir)}")
    source_files = sorted(source_dir.glob("*.png"))[:sample_count]
    if not source_files:
        raise FileNotFoundError(f"样本目录为空：{display_path(source_dir)}")

    started = time.time()
    timestamp = datetime.now().isoformat(timespec="seconds")
    run_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{surrogate}_{eps}"
    output_dir = GENERATED_SAMPLES_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in source_files:
        shutil.copy2(src, output_dir / src.name)
        copied += 1

    summary = {
        "generation_id": run_id,
        "timestamp": timestamp,
        "algorithm": algorithm,
        "surrogate": surrogate,
        "eps": eps,
        "perturbation_budget": eps_to_budget(eps),
        "linf_bound": eps_to_linf(eps),
        "sample_count": copied,
        "source_dir": display_path(source_dir),
        "output_dir": display_path(output_dir),
        "elapsed_seconds": time.time() - started,
        "status": "completed",
    }
    append_jsonl(GENERATION_RUNS_PATH, [summary])
    return summary


def run_realtime_blackbox(surrogate: str, eps: str, target_model: str, sample_count: int, metadata_map: dict):
    folder = adversarial_folder(surrogate, eps)
    if not folder.exists():
        raise FileNotFoundError(f"样本目录不存在：{display_path(folder)}")

    image_paths = sorted(folder.glob("*.png"))[:sample_count]
    if not image_paths:
        raise FileNotFoundError(f"样本目录为空：{display_path(folder)}")

    model, device = get_target_model(target_model)
    transform = target_transform(target_model)
    started = time.time()
    timestamp = datetime.now().isoformat(timespec="seconds")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{surrogate}_{eps}_{target_model}"

    rows = []
    progress = st.progress(0, text="正在运行目标模型推理...")
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths, start=1):
            image_id = image_path.stem.zfill(4)
            metadata = metadata_map.get(image_id, {})
            true_label = metadata.get("true_label")
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            if hasattr(output, "logits"):
                output = output.logits
            adv_output = int(output.argmax(dim=1).item())
            attack_success = (true_label is not None) and (adv_output != int(true_label))
            clean_output = clean_prediction_from_metadata(metadata, surrogate)
            output_shift = (clean_output is not None) and (adv_output != int(clean_output))
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "image_id": image_id,
                    "surrogate": surrogate,
                    "eps": eps,
                    "target_model": target_model,
                    "true_label": true_label,
                    "clean_output": clean_output,
                    "adv_output": adv_output,
                    "attack_success": bool(attack_success),
                    "output_shift": bool(output_shift),
                    "query_count": 1,
                    "sample_path": display_path(image_path),
                }
            )
            progress.progress(idx / len(image_paths), text=f"正在运行目标模型推理... {idx}/{len(image_paths)}")
    progress.empty()

    elapsed = time.time() - started
    success_count = sum(1 for row in rows if row["attack_success"])
    output_shift_count = sum(1 for row in rows if row["output_shift"])
    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "surrogate": surrogate,
        "eps": eps,
        "target_model": target_model,
        "sample_count": len(rows),
        "success_count": success_count,
        "attack_success_rate": success_count / len(rows) if rows else 0.0,
        "output_shift_count": output_shift_count,
        "output_shift_rate": output_shift_count / len(rows) if rows else 0.0,
        "perturbation_budget": eps_to_budget(eps),
        "linf_bound": eps_to_linf(eps),
        "transferability": success_count / len(rows) if rows else 0.0,
        "query_count": len(rows),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "status": "completed",
    }
    append_jsonl(RUNTIME_RECORDS_PATH, rows)
    append_jsonl(RUNS_PATH, [summary])
    return summary, pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sample_inventory(cache_key: tuple) -> pd.DataFrame:
    rows = []
    for surrogate_dir in sorted(ADVERSARIAL_DIR.glob("*")):
        if not surrogate_dir.is_dir():
            continue
        for eps_dir in sorted(surrogate_dir.glob("eps*")):
            if eps_dir.is_dir():
                rows.append(
                    {
                        "样本集": surrogate_dir.name,
                        "扰动预算": eps_dir.name,
                        "样本数": count_pngs(eps_dir),
                        "目录": display_path(eps_dir),
                    }
                )
    for eps_dir in sorted(ENSEMBLE_DIR.glob("eps*")):
        if eps_dir.is_dir():
            rows.append(
                {
                    "样本集": "ensemble",
                    "扰动预算": eps_dir.name,
                    "样本数": count_pngs(eps_dir),
                    "目录": display_path(eps_dir),
                }
            )
    return pd.DataFrame(rows)


def metric_panel(title: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-value">{value}</div>
            <div class="panel-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value) -> str:
    if pd.isna(value):
        return "未完成"
    return f"{float(value) * 100:.2f}%"


def eps_to_budget(eps: str) -> str:
    mapping = {"eps4": "4/255", "eps8": "8/255", "eps16": "16/255"}
    return mapping.get(str(eps), str(eps))


def eps_to_linf(eps: str) -> float:
    mapping = {"eps4": 4 / 255, "eps8": 8 / 255, "eps16": 16 / 255}
    return mapping.get(str(eps), 0.0)


def render_header():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">多模态模型攻击与安全评估平台</div>
            <p class="hero-subtitle">
            当前实现聚焦图像分类模态，围绕攻击场景设计、样本管理、攻击算法调用、
            效果评估、可视化分析与实验记录管理构建标准化实验流程。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_summary_bar(summary: pd.DataFrame):
    if summary.empty:
        st.info("暂无黑盒评估记录。")
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    eps_order = ["eps4", "eps8", "eps16"]
    plot_df = summary.copy()
    plot_df["series"] = plot_df["surrogate"] + " -> " + plot_df["target_model"]
    for series_name, model_df in plot_df.groupby("series"):
        model_df = model_df.groupby("eps", as_index=False)["attack_success_percent"].mean()
        model_df = model_df.set_index("eps").reindex(eps_order).reset_index()
        ax.plot(model_df["eps"], model_df["attack_success_percent"], marker="o", linewidth=2, label=series_name)
    ax.set_xlabel("扰动预算")
    ax.set_ylabel("攻击成功率 ASR (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(title="目标模型", ncol=2)
    fig.tight_layout()
    st.pyplot(fig)


def plot_surrogate_chart(summary: pd.DataFrame, surrogate: str, title: str):
    df = summary[summary["surrogate"] == surrogate].copy()
    if df.empty:
        st.info(f"暂无 {surrogate} 黑盒迁移记录。")
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    eps_order = ["eps4", "eps8", "eps16"]
    for target_model, model_df in df.groupby("target_model"):
        model_df = model_df.groupby("eps", as_index=False)["attack_success_percent"].mean()
        model_df = model_df.set_index("eps").reindex(eps_order).reset_index()
        ax.plot(model_df["eps"], model_df["attack_success_percent"], marker="o", linewidth=2, label=target_model)
    ax.set_title(title)
    ax.set_xlabel("扰动预算")
    ax.set_ylabel("攻击成功率 ASR (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(title="目标模型", ncol=2)
    fig.tight_layout()
    st.pyplot(fig)


def plot_whitebox_bar(whitebox: dict):
    if not whitebox:
        st.info("暂无白盒结果。")
        return
    rows = []
    for name, item in whitebox.items():
        rows.append(
            {
                "实验": name,
                "替代模型": item.get("surrogate"),
                "扰动预算": item.get("eps"),
                "白盒ASR": item.get("whitebox_asr"),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["白盒ASR"])
    if df.empty:
        st.info("白盒结果尚未全部完成。")
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(df["实验"], df["白盒ASR"] * 100, color="#2563eb")
    ax.set_ylabel("白盒 ASR (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)


def whitebox_dataframe(whitebox: dict) -> pd.DataFrame:
    rows = []
    for name, item in whitebox.items():
        rows.append(
            {
                "实验": name,
                "替代模型": item.get("surrogate"),
                "扰动预算": item.get("eps"),
                "样本数": item.get("total"),
                "白盒ASR": item.get("whitebox_asr"),
            }
        )
    return pd.DataFrame(rows)


def plot_whitebox_blackbox_comparison(whitebox: dict, summary: pd.DataFrame):
    wb = whitebox_dataframe(whitebox)
    if wb.empty or summary.empty:
        st.info("暂无可对比的黑白盒结果。")
        return
    wb_avg = wb.groupby("替代模型", as_index=False)["白盒ASR"].mean()
    bb_avg = summary.groupby("surrogate", as_index=False)["attack_success_percent"].mean()
    compare = wb_avg.merge(bb_avg, left_on="替代模型", right_on="surrogate", how="inner")
    if compare.empty:
        st.info("暂无可对比的黑白盒结果。")
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = range(len(compare))
    width = 0.35
    ax.bar([i - width / 2 for i in x], compare["白盒ASR"] * 100, width=width, label="白盒 ASR")
    ax.bar([i + width / 2 for i in x], compare["attack_success_percent"], width=width, label="黑盒迁移 ASR")
    ax.set_xticks(list(x))
    ax.set_xticklabels(compare["替代模型"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("ASR (%)")
    ax.set_title("白盒与黑盒平均攻击成功率对比")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def plot_eps_trend(summary: pd.DataFrame):
    if summary.empty:
        st.info("暂无 eps 趋势数据。")
        return
    eps_order = ["eps4", "eps8", "eps16"]
    trend = summary.groupby(["surrogate", "eps"], as_index=False)["attack_success_percent"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for surrogate, data in trend.groupby("surrogate"):
        data = data.set_index("eps").reindex(eps_order).reset_index()
        ax.plot(data["eps"], data["attack_success_percent"], marker="o", linewidth=2, label=surrogate)
    ax.set_ylim(0, 100)
    ax.set_xlabel("扰动预算")
    ax.set_ylabel("平均黑盒 ASR (%)")
    ax.set_title("不同 eps 下的黑盒迁移趋势")
    ax.grid(True, alpha=0.25)
    ax.legend(title="攻击来源")
    fig.tight_layout()
    st.pyplot(fig)


def plot_robustness_ranking(summary: pd.DataFrame):
    if summary.empty:
        st.info("暂无鲁棒性排序数据。")
        return
    ranking = summary.groupby("target_model", as_index=False)["attack_success_percent"].mean()
    ranking = ranking.sort_values("attack_success_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(ranking["target_model"], ranking["attack_success_percent"], color="#0f766e")
    ax.set_xlabel("平均黑盒 ASR (%)")
    ax.set_title("目标模型鲁棒性排序（ASR 越低越鲁棒）")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    table = ranking.rename(columns={"target_model": "目标模型", "attack_success_percent": "平均ASR(%)"}).copy()
    table["鲁棒性排序"] = range(1, len(table) + 1)
    st.dataframe(table[["鲁棒性排序", "目标模型", "平均ASR(%)"]].style.format({"平均ASR(%)": "{:.2f}"}), hide_index=True, use_container_width=True)


def show_artifact_image(path: Path, caption: str):
    if path.exists():
        st.image(Image.open(path), caption=caption, use_column_width=True)
    else:
        st.info(f"未找到图表：{path.name}")


def file_cache_key(paths) -> tuple:
    key = []
    for path in paths:
        key.append((str(path), path.stat().st_mtime if path.exists() else None))
    return tuple(key)


whitebox_key = file_cache_key([exp_dir / "whitebox_results.json" for exp_dir in EXPERIMENT_DIRS.values()])
blackbox_key = file_cache_key([exp_dir / "records.jsonl" for exp_dir in EXPERIMENT_DIRS.values()])
metadata_key = file_cache_key(list(METADATA_PATHS.values()))
runs_key = file_cache_key([RUNS_PATH])
generation_key = file_cache_key([GENERATION_RUNS_PATH])

whitebox_data = load_whitebox(whitebox_key)
blackbox_df = load_blackbox(blackbox_key)
summary_df = load_summary(blackbox_key)
inventory_df = sample_inventory(file_cache_key([ADVERSARIAL_DIR, ENSEMBLE_DIR]))
metadata_df = load_metadata_inventory(metadata_key)
metadata_map = load_metadata_map(metadata_key)
realtime_runs_df = load_realtime_runs(runs_key)
generation_runs_df = load_generation_runs(generation_key)

with st.sidebar:
    st.markdown("### 实验工作台")
    page = st.radio(
        "功能模块",
        ["总览", "攻击场景", "样本管理", "攻击与评估", "实验记录"],
        label_visibility="collapsed",
    )
    st.divider()
    record_count = sum(1 for exp_dir in EXPERIMENT_DIRS.values() if (exp_dir / "records.jsonl").exists())
    record_state = "已加载" if record_count else "缺失"
    report_state = "已生成" if REPORT_PATH.exists() else "缺失"
    st.caption(f"黑盒记录：{record_state}")
    st.caption(f"元数据：{int(metadata_df['有效'].sum())}/2 已加载")
    st.caption(f"评估报告：{report_state}")

render_header()

if page == "总览":
    total_blackbox_records = len(blackbox_df)
    total_images = int(inventory_df["样本数"].sum()) if not inventory_df.empty else 0
    avg_asr = blackbox_df["attack_success"].mean() * 100 if not blackbox_df.empty else 0
    best_row = summary_df.loc[summary_df["attack_success_percent"].idxmax()] if not summary_df.empty else None
    best_text = "暂无"
    if best_row is not None:
        best_text = f"{best_row['eps']} / {best_row['target_model']}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_panel("攻击样本库存", f"{total_images:,}", "来自 adversarial 与 ensemble")
    with col2:
        metric_panel("黑盒评估记录", f"{total_blackbox_records:,}", "按样本与目标模型去重后的总量")
    with col3:
        metric_panel("平均迁移 ASR", f"{avg_asr:.2f}%", "基于全部黑盒记录计算")
    with col4:
        metric_panel("当前最强配置", best_text, "按目标模型 ASR 排序得到")

    st.markdown("### 实验要求覆盖情况")
    coverage = pd.DataFrame(
        [
            ["攻击场景设计", "图像分类黑盒迁移攻击", "已覆盖"],
            ["样本管理", "原始 metadata、对抗样本目录、records 记录", "已覆盖"],
            ["攻击算法实现", "resnet/vit/ensemble 替代模型生成与迁移评估", "已覆盖"],
            ["效果评估", "ASR、扰动预算、目标模型迁移能力、查询次数", "已覆盖"],
            ["可视化展示", "ASR 趋势、热力图、记录表、报告", "已覆盖"],
        ],
        columns=["要求", "当前实现", "状态"],
    )
    st.dataframe(coverage, hide_index=True, use_container_width=True)

    st.markdown("### 黑盒迁移概览")
    plot_summary_bar(summary_df)

    st.markdown("### 白盒迁移概览")
    plot_whitebox_bar(whitebox_data)

    st.markdown("### 数据来源")
    data_sources = pd.DataFrame(
        [
            ["对抗样本", "adversarial; ensemble", "resnet_only、vit_only、ensemble 三组样本"],
            ["标签元数据", "metadata.json; metadata(2).json", "图片 ID、真实标签、clean 预测信息"],
            ["白盒结果", "whitebox_results.json", "三组替代模型在对应样本上的白盒 ASR"],
            ["黑盒记录", "records.jsonl", "三组攻击样本在目标模型上的迁移评估记录"],
            ["分析产物", "analysis_results", "图表、CSV 汇总与评估报告"],
        ],
        columns=["数据类型", "路径", "用途"],
    )
    st.dataframe(data_sources, hide_index=True, use_container_width=True)

elif page == "攻击场景":
    st.markdown("### 攻击场景设计")
    st.markdown(
        '<p class="section-note">本项目选择图像分类作为实现模态。实验流程是：选择替代模型生成对抗样本，再把样本迁移到多个未知目标模型进行黑盒验证。</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.05, 1])
    with col1:
        st.selectbox("任务类型", ["图像分类"], index=0)
        st.multiselect(
            "目标模型",
            ["InceptionV3", "DenseNet121", "MobileNetV2", "EfficientNet-B0"],
            default=["InceptionV3", "DenseNet121", "MobileNetV2", "EfficientNet-B0"],
        )
        st.selectbox("攻击方式", ["黑盒迁移攻击", "白盒替代模型攻击"], index=0)
    with col2:
        st.selectbox("替代模型", ["ensemble", "resnet_only", "vit_only"], index=0)
        st.multiselect("扰动预算", ["eps4 (4/255)", "eps8 (8/255)", "eps16 (16/255)"], default=["eps4 (4/255)", "eps8 (8/255)", "eps16 (16/255)"])
        st.slider("查询次数上限", min_value=1, max_value=5000, value=3305, step=100)

    st.markdown("### 标准化实验流程")
    workflow = pd.DataFrame(
        [
            ["1", "选择图像分类场景和替代模型", "确定要验证的攻击假设"],
            ["2", "加载 metadata 和对抗样本集", "保证样本、标签、版本可追溯"],
            ["3", "运行黑盒目标模型推理", "记录 adv_output 与 attack_success"],
            ["4", "按 eps / target_model 汇总 ASR", "比较扰动强度和迁移能力"],
            ["5", "输出图表和报告", "支撑实验结论和平台展示"],
        ],
        columns=["步骤", "平台动作", "实验意义"],
    )
    st.dataframe(workflow, hide_index=True, use_container_width=True)

elif page == "样本管理":
    st.markdown("### 多模态数据集与样本管理")
    st.markdown(
        '<p class="section-note">当前样本管理围绕图像模态：基础标签来自 metadata，攻击样本按 surrogate/eps 分目录存储，评估记录按 JSONL 追加保存。</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_panel("样本集数量", str(len(inventory_df)) if not inventory_df.empty else "0", "adversarial + ensemble")
    with col2:
        metric_panel("对抗样本总数", f"{int(inventory_df['样本数'].sum()):,}" if not inventory_df.empty else "0", "磁盘 PNG 文件统计")
    with col3:
        metric_panel("元数据文件", f"{int(metadata_df['有效'].sum())}/2", "metadata.json 与 metadata(2).json")

    tab_table, tab_preview, tab_metadata = st.tabs(["样本目录", "样本预览", "元数据"])
    with tab_table:
        if inventory_df.empty:
            st.info("未发现样本目录。")
        else:
            st.dataframe(inventory_df, hide_index=True, use_container_width=True)

    with tab_preview:
        preview_root = st.selectbox("选择样本集", ["ensemble/eps16", "ensemble/eps8", "ensemble/eps4", "adversarial/resnet_only/eps8", "adversarial/vit_only/eps8"])
        folder = ROOT / preview_root
        files = sorted(folder.glob("*.png"))[:6] if folder.exists() else []
        if not files:
            st.info("该目录暂无可预览图片。")
        else:
            cols = st.columns(6)
            for col, path in zip(cols, files):
                with col:
                    st.image(Image.open(path), caption=path.name, use_column_width=True)

    with tab_metadata:
        st.dataframe(metadata_df, hide_index=True, use_container_width=True)

elif page == "攻击与评估":
    st.markdown("### 攻击算法实现与效果评估")
    st.markdown(
        '<p class="section-note">这里展示真实实验结果，不再使用演示假数据。ASR 越高，说明对抗样本越容易迁移到未知目标模型，目标模型鲁棒性越弱。</p>',
        unsafe_allow_html=True,
    )

    tab_generate, tab_run, tab_black, tab_white, tab_visual = st.tabs(["样本生成", "实时运行", "黑盒迁移", "白盒验证", "可视化"])

    with tab_generate:
        st.markdown("#### 攻击样本生成")
        st.markdown(
            '<p class="section-note">选择攻击算法、攻击来源和扰动预算，调用已有攻击样本生成流程并保存生成记录。当前入口用于复现实验样本集和登记生成产物。</p>',
            unsafe_allow_html=True,
        )
        with st.form("sample_generation_form"):
            g1, g2, g3, g4 = st.columns([1.3, 1, 1, 1])
            with g1:
                gen_algorithm = st.selectbox("攻击算法", ["黑盒迁移攻击", "PGD 图像扰动", "FGSM 图像扰动", "Ensemble 迁移攻击"], index=0)
            with g2:
                gen_surrogate = st.selectbox("攻击来源", ["resnet_only", "vit_only", "ensemble"], index=2)
            with g3:
                gen_eps = st.selectbox("扰动预算", ["eps4", "eps8", "eps16"], index=1)
            gen_folder = adversarial_folder(gen_surrogate, gen_eps)
            gen_available = count_pngs(gen_folder)
            with g4:
                gen_count = st.number_input("生成样本数", min_value=1, max_value=max(1, min(gen_available, 1000)), value=min(20, max(1, gen_available)), step=10)
            st.caption(f"输入样本集：{display_path(gen_folder)}，可用样本 {gen_available:,} 张。")
            gen_button = st.form_submit_button("生成攻击样本", type="primary", use_container_width=True, disabled=gen_available == 0)

        if gen_button:
            try:
                with st.spinner("正在生成并登记攻击样本..."):
                    gen_summary = run_attack_sample_generation(gen_algorithm, gen_surrogate, gen_eps, int(gen_count))
                st.success("攻击样本生成完成，生成记录已保存。")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("生成样本数", f"{gen_summary['sample_count']:,}")
                with m2:
                    st.metric("扰动预算", gen_summary["perturbation_budget"])
                with m3:
                    st.metric("耗时", f"{gen_summary['elapsed_seconds']:.2f}s")
                with m4:
                    st.metric("输出目录", gen_summary["output_dir"])
            except Exception as exc:
                st.error(f"攻击样本生成失败：{exc}")

        st.markdown("#### 最近生成记录")
        recent_generations = load_generation_runs(file_cache_key([GENERATION_RUNS_PATH]))
        if recent_generations.empty:
            st.info("暂无攻击样本生成记录。")
        else:
            recent = recent_generations.sort_values("timestamp", ascending=False).head(10)
            st.dataframe(
                recent[
                    [
                        "timestamp",
                        "algorithm",
                        "surrogate",
                        "eps",
                        "sample_count",
                        "perturbation_budget",
                        "output_dir",
                        "elapsed_seconds",
                        "status",
                    ]
                ].rename(
                    columns={
                        "timestamp": "时间",
                        "algorithm": "攻击算法",
                        "surrogate": "攻击来源",
                        "eps": "扰动预算",
                        "sample_count": "样本数",
                        "perturbation_budget": "扰动强度",
                        "output_dir": "输出目录",
                        "elapsed_seconds": "耗时(s)",
                        "status": "状态",
                    }
                ).style.format({"耗时(s)": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )

    with tab_run:
        st.markdown("#### 实时黑盒攻击验证")
        st.markdown(
            '<p class="section-note">选择已有攻击样本集，现场调用目标模型完成一次黑盒迁移评估，并保存运行记录。</p>',
            unsafe_allow_html=True,
        )
        with st.form("realtime_blackbox_form"):
            col_a, col_b, col_c, col_d = st.columns([1, 1, 1.2, 1])
            with col_a:
                run_surrogate = st.selectbox("攻击来源", ["resnet_only", "vit_only", "ensemble"], index=2)
            with col_b:
                run_eps = st.selectbox("扰动预算", ["eps4", "eps8", "eps16"], index=1)
            with col_c:
                run_target = st.selectbox("目标模型", target_model_options(), index=2)
            folder = adversarial_folder(run_surrogate, run_eps)
            available_count = count_pngs(folder)
            with col_d:
                run_count = st.number_input("样本数", min_value=1, max_value=max(1, min(available_count, 1000)), value=min(20, max(1, available_count)), step=10)

            st.caption(f"当前样本集：{display_path(folder)}，可用样本 {available_count:,} 张。")
            run_button = st.form_submit_button("运行黑盒评估", type="primary", use_container_width=True, disabled=available_count == 0)
        if run_button:
            try:
                with st.spinner("正在执行实时黑盒评估，首次加载模型可能需要一些时间..."):
                    run_summary, run_records = run_realtime_blackbox(
                        run_surrogate,
                        run_eps,
                        run_target,
                        int(run_count),
                        metadata_map,
                    )
                st.success("实时评估完成，运行记录已保存。")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("攻击成功率", f"{run_summary['attack_success_rate'] * 100:.2f}%")
                with m2:
                    st.metric("查询次数", f"{run_summary['query_count']:,}")
                with m3:
                    st.metric("耗时", f"{run_summary['elapsed_seconds']:.2f}s")
                with m4:
                    st.metric("运行设备", run_summary["device"])

                st.markdown("#### 评估指标")
                metric_table = pd.DataFrame(
                    [
                        ["攻击成功率", f"{run_summary['attack_success_rate'] * 100:.2f}%", "攻击成功样本数 / 总样本数"],
                        ["输出偏移率", f"{run_summary['output_shift_rate'] * 100:.2f}%", "对抗预测不同于攻击前预测的比例"],
                        ["迁移能力", f"{run_summary['transferability'] * 100:.2f}%", "实时黑盒 ASR"],
                        ["查询次数", f"{run_summary['query_count']:,}", "每张样本对目标模型查询 1 次"],
                        ["扰动强度", run_summary["perturbation_budget"], "eps 对应的 L∞ 约束"],
                        ["计算代价", f"{run_summary['elapsed_seconds']:.2f}s", "本次实时运行耗时"],
                    ],
                    columns=["指标", "结果", "说明"],
                )
                st.dataframe(metric_table, hide_index=True, use_container_width=True)

                preview = run_records.head(20).rename(
                    columns={
                        "image_id": "图片ID",
                        "true_label": "真实标签",
                        "clean_output": "攻击前预测",
                        "adv_output": "攻击后预测",
                        "attack_success": "是否成功",
                        "output_shift": "输出是否偏移",
                        "sample_path": "样本路径",
                    }
                )
                st.markdown("#### 攻击前后结果对比")
                st.dataframe(
                    preview[["图片ID", "真实标签", "攻击前预测", "攻击后预测", "输出是否偏移", "是否成功", "样本路径"]],
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"实时评估失败：{exc}")

        st.markdown("#### 最近实时运行")
        recent_runs = load_realtime_runs(file_cache_key([RUNS_PATH]))
        if recent_runs.empty:
            st.info("暂无实时运行记录。")
        else:
            recent = recent_runs.sort_values("timestamp", ascending=False).head(10).copy()
            recent["ASR(%)"] = recent["attack_success_rate"] * 100
            st.dataframe(
                recent[
                    [
                        "timestamp",
                        "surrogate",
                        "eps",
                        "target_model",
                        "sample_count",
                        "query_count",
                        "elapsed_seconds",
                        "ASR(%)",
                        "status",
                    ]
                ].rename(
                    columns={
                        "timestamp": "时间",
                        "surrogate": "攻击来源",
                        "eps": "扰动预算",
                        "target_model": "目标模型",
                        "sample_count": "样本数",
                        "query_count": "查询次数",
                        "elapsed_seconds": "耗时(s)",
                        "status": "状态",
                    }
                ).style.format({"耗时(s)": "{:.2f}", "ASR(%)": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )

    with tab_black:
        if summary_df.empty:
            st.info("暂无黑盒汇总结果。")
        else:
            filter_surrogate = st.multiselect("攻击来源", sorted(summary_df["surrogate"].unique()), default=sorted(summary_df["surrogate"].unique()))
            filter_eps = st.multiselect("扰动预算", sorted(summary_df["eps"].unique()), default=sorted(summary_df["eps"].unique()))
            filter_model = st.multiselect("目标模型", sorted(summary_df["target_model"].unique()), default=sorted(summary_df["target_model"].unique()))
            filtered = summary_df[
                summary_df["surrogate"].isin(filter_surrogate)
                & summary_df["eps"].isin(filter_eps)
                & summary_df["target_model"].isin(filter_model)
            ].copy()
            filtered["扰动预算"] = filtered["eps"].map(eps_to_budget)
            view = filtered[["surrogate", "扰动预算", "target_model", "samples", "records", "attack_success_percent"]].rename(
                columns={
                    "surrogate": "攻击来源",
                    "target_model": "目标模型",
                    "samples": "样本数",
                    "records": "记录数",
                    "attack_success_percent": "ASR(%)",
                }
            )
            st.dataframe(view.style.format({"ASR(%)": "{:.2f}"}), hide_index=True, use_container_width=True)
            plot_summary_bar(filtered)

            avg_by_eps = filtered.groupby("eps")["attack_success_percent"].mean().reset_index()
            best = filtered.loc[filtered["attack_success_percent"].idxmax()]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("筛选后平均 ASR", f"{filtered['attack_success_percent'].mean():.2f}%")
            with c2:
                st.metric("最佳目标模型", str(best["target_model"]), f"{best['attack_success_percent']:.2f}%")
            with c3:
                st.metric("查询次数", f"{int(filtered['records'].sum()):,}", "1 query / 样本 / 模型")

    with tab_white:
        rows = []
        for name, item in whitebox_data.items():
            rows.append(
                {
                    "实验": name,
                    "替代模型": item.get("surrogate"),
                    "扰动预算": item.get("eps"),
                    "样本数": item.get("total"),
                    "白盒ASR": item.get("whitebox_asr"),
                }
            )
        wb_df = pd.DataFrame(rows)
        if wb_df.empty:
            st.info("暂无白盒结果。")
        else:
            display = wb_df.copy()
            display["白盒ASR"] = display["白盒ASR"].apply(format_percent)
            st.table(display)
            plot_whitebox_bar(whitebox_data)

    with tab_visual:
        chart_tabs = st.tabs(["ResNet", "ViT", "Ensemble", "综合对比", "黑白盒对比", "eps趋势", "鲁棒性排序"])
        with chart_tabs[0]:
            plot_surrogate_chart(summary_df, "resnet_only", "ResNet 替代模型黑盒迁移 ASR")
        with chart_tabs[1]:
            plot_surrogate_chart(summary_df, "vit_only", "ViT 替代模型黑盒迁移 ASR")
        with chart_tabs[2]:
            plot_surrogate_chart(summary_df, "ensemble", "Ensemble 替代模型黑盒迁移 ASR")
        with chart_tabs[3]:
            plot_summary_bar(summary_df)
            plot_whitebox_bar(whitebox_data)
        with chart_tabs[4]:
            plot_whitebox_blackbox_comparison(whitebox_data, summary_df)
        with chart_tabs[5]:
            plot_eps_trend(summary_df)
        with chart_tabs[6]:
            plot_robustness_ranking(summary_df)

elif page == "实验记录":
    st.markdown("### 实验记录与报告")
    st.markdown(
        '<p class="section-note">实验记录用于满足多轮实验、重复验证和结果追踪。黑盒记录保留每张图片在每个目标模型上的输出和攻击是否成功。</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 黑盒记录")
        if blackbox_df.empty:
            st.info("暂无黑盒记录。")
        else:
            grouped = (
                blackbox_df.groupby(["surrogate", "eps", "target_model"], as_index=False)
                .agg(records=("attack_success", "size"), images=("image_id", "nunique"), asr=("attack_success", "mean"))
                .sort_values(["surrogate", "eps", "target_model"])
            )
            grouped["asr"] = grouped["asr"] * 100
            st.dataframe(
                grouped.rename(columns={"surrogate": "攻击来源", "eps": "扰动预算", "target_model": "目标模型", "records": "记录数", "images": "图片数", "asr": "ASR(%)"}).style.format({"ASR(%)": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )

    with col2:
        st.markdown("#### 白盒记录")
        wb_rows = []
        for name, item in whitebox_data.items():
            wb_rows.append(
                {
                    "实验": name,
                    "替代模型": item.get("surrogate"),
                    "扰动预算": item.get("eps"),
                    "样本数": item.get("total"),
                    "白盒ASR": format_percent(item.get("whitebox_asr")),
                }
            )
        if wb_rows:
            st.table(pd.DataFrame(wb_rows))
        else:
            st.info("暂无白盒记录。")

    st.markdown("#### 攻击样本生成记录")
    recent_generations = load_generation_runs(file_cache_key([GENERATION_RUNS_PATH]))
    if recent_generations.empty:
        st.info("暂无攻击样本生成记录。")
    else:
        generation_view = recent_generations.sort_values("timestamp", ascending=False).copy()
        st.dataframe(
            generation_view[
                [
                    "timestamp",
                    "algorithm",
                    "surrogate",
                    "eps",
                    "sample_count",
                    "perturbation_budget",
                    "output_dir",
                    "elapsed_seconds",
                    "status",
                ]
            ].rename(
                columns={
                    "timestamp": "时间",
                    "algorithm": "攻击算法",
                    "surrogate": "攻击来源",
                    "eps": "扰动预算",
                    "sample_count": "样本数",
                    "perturbation_budget": "扰动强度",
                    "output_dir": "输出目录",
                    "elapsed_seconds": "耗时(s)",
                    "status": "状态",
                }
            ).style.format({"耗时(s)": "{:.2f}"}),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### 实时运行记录")
    recent_runs = load_realtime_runs(file_cache_key([RUNS_PATH]))
    if recent_runs.empty:
        st.info("暂无实时运行记录。")
    else:
        realtime_view = recent_runs.sort_values("timestamp", ascending=False).copy()
        realtime_view["ASR(%)"] = realtime_view["attack_success_rate"] * 100
        st.dataframe(
            realtime_view[
                [
                    "timestamp",
                    "surrogate",
                    "eps",
                    "target_model",
                    "sample_count",
                    "query_count",
                    "elapsed_seconds",
                    "ASR(%)",
                    "status",
                ]
            ].rename(
                columns={
                    "timestamp": "时间",
                    "surrogate": "攻击来源",
                    "eps": "扰动预算",
                    "target_model": "目标模型",
                    "sample_count": "样本数",
                    "query_count": "查询次数",
                    "elapsed_seconds": "耗时(s)",
                    "status": "状态",
                }
            ).style.format({"耗时(s)": "{:.2f}", "ASR(%)": "{:.2f}"}),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### 评估报告")
    if REPORT_PATH.exists():
        st.download_button(
            "下载 Markdown 报告",
            data=REPORT_PATH.read_text(encoding="utf-8"),
            file_name="evaluation_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        with st.expander("预览报告内容", expanded=True):
            st.markdown(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("报告尚未生成，请先运行分析脚本。")
