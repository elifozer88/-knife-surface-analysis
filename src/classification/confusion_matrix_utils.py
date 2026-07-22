import os
import matplotlib.pyplot as plt


def print_terminal_confusion_matrix(cm, class_names):
    """Print a simple terminal-friendly visual confusion matrix using unicode blocks."""
    if cm.ndim != 2:
        print("⚠ ERROR: Confusion matrix must be 2-dimensional.")
        return

    print("\n--- CONFUSION MATRIX (Terminal View) ---")
    max_value = float(cm.max()) if cm.size else 1.0
    if max_value <= 0:
        max_value = 1.0

    header = "    " + "  ".join(f"{name:<10}" for name in class_names)
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        # class_names listesinin dışına taşmayı önlemek için güvenli indeksleme
        class_label = class_names[i] if i < len(class_names) else f"Class {i}"
        row_text = f"{class_label:<4}"
        for value in row:
            intensity = (
                int(round((value / max_value) * 9)) if max_value > 0 else 0
            )
            intensity = max(1, min(9, intensity)) if value > 0 else 0
            block = "█" * intensity
            row_text += f" {block:<10}"
        print(f"{row_text} [{', '.join(map(str, row.tolist()))}]")


def save_confusion_matrix_image(cm, class_names, output_path):
    """Save a colored confusion matrix image to disk with automated directory generation."""
    # Sıkı Eleme: Hedef dizin yoksa otomatik oluştur, böylece çökme yaşanmaz
    target_dir = os.path.dirname(output_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, pad=15, fontweight="bold")

    # Dinamik ticks ayarları
    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=15, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)

    # Hücre içi sayıların kontrast kontrolü
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    try:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"✓ Confusion Matrix image saved successfully to '{output_path}'")
    except OSError as ex:
        print(
            f"WARNING: Could not save confusion matrix image to {output_path}: {ex}"
        )
    finally:
        plt.close(fig)