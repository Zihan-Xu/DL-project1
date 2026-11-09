"""Build a plain course report from the submitted version's frozen results.

Usage: python report_source.py --data-root . --output report.pdf
Uses system Times New Roman when available, otherwise PDF's standard Times fonts.
"""

import argparse
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build(data_root, output):
    font_dir = Path("C:/Windows/Fonts")
    if (font_dir / "times.ttf").exists():
        for name, filename in [
            ("ReportTimes", "times.ttf"),
            ("ReportTimes-Bold", "timesbd.ttf"),
            ("ReportTimes-Italic", "timesi.ttf"),
        ]:
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        pdfmetrics.registerFontFamily(
            "ReportTimes",
            normal="ReportTimes",
            bold="ReportTimes-Bold",
            italic="ReportTimes-Italic",
            boldItalic="ReportTimes-Bold",
        )
        regular, bold = "ReportTimes", "ReportTimes-Bold"
    else:
        regular, bold = "Times-Roman", "Times-Bold"
    body = ParagraphStyle(
        "Body",
        fontName=regular,
        fontSize=11,
        leading=14,
        spaceAfter=7,
        alignment=TA_LEFT,
    )
    title = ParagraphStyle(
        "Title",
        parent=body,
        fontName=bold,
        fontSize=16,
        leading=19,
        spaceAfter=8,
        alignment=TA_CENTER,
    )
    author = ParagraphStyle(
        "Author",
        parent=body,
        fontSize=11,
        leading=14,
        spaceAfter=5,
        alignment=TA_CENTER,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName=bold,
        fontSize=12,
        leading=15,
        spaceBefore=7,
        spaceAfter=6,
        keepWithNext=True,
    )
    small = ParagraphStyle("Small", parent=body, fontSize=9.5, leading=12, spaceAfter=7)
    cell = ParagraphStyle("Cell", parent=body, fontSize=10, leading=12, spaceAfter=0)
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def h(text):
        p(text, heading)

    def table(headers, rows, widths):
        values = [[Paragraph("<b>" + str(x) + "</b>", cell) for x in headers]]
        values.extend([[Paragraph(str(x), cell) for x in row] for row in rows])
        t = Table(values, colWidths=widths, repeatRows=1, hAlign="CENTER")
        t.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.65, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.65, colors.black),
                ]
            )
        )
        story.extend([t, Spacer(1, 6)])

    def page_number(canvas, doc):
        canvas.setFont(regular, 9)
        canvas.drawCentredString(letter[0] / 2, 35, str(doc.page))

    p("CIFAR-10 Classification with a Modified ResNet", title)
    p(
        "Zihan Xu<br/>Kaggle team: Zihan Xu<br/>Deep Learning Project 1, Fall 2026",
        author,
    )
    p(
        '<link href="https://github.com/Zihan-Xu/DL-project1" color="black">https://github.com/Zihan-Xu/DL-project1</link>',
        author,
    )

    h("1. Introduction")
    p(
        "I trained a modified ResNet-20 for the course CIFAR-10 competition [7]. The final model has 2,436,346 trainable parameters and achieved 87.90% accuracy on the Kaggle public leaderboard. It was trained from scratch on all 50,000 permitted training images. The selected training configuration reached 96.04% mean validation accuracy in a three-seed study before the full-data fit. These are different evaluations: the validation result comes from models trained on 40,000 images, while the public score comes from the final submitted model."
    )

    h("2. Model and training")
    p(
        "The network follows the CIFAR ResNet design [1], with three residual stages and three blocks per stage. Each block contains two 3 x 3 convolutions with batch normalization and ReLU, plus a shortcut. I used 48, 96 and 192 channels across the stages, followed by global average pooling and a ten-class linear layer. This adds capacity over the narrow baseline while keeping the parameter count below the five-million limit."
    )
    p(
        "At stage transitions, a fixed 3 x 3 binomial BlurPool filter downsamples both the residual and shortcut paths [2]. The shortcut also uses a learned 1 x 1 projection to match channel counts. BlurPool adds no trainable parameters, although it increases computation. A single forward pass costs about 436.1 million multiply-accumulate operations."
    )
    p(
        "Development used a fixed, class-stratified split of 40,000 training images, 5,000 validation images and 5,000 historical local-test images. Images were normalized using statistics computed from the training portion. Checkpoints were selected by raw validation accuracy, with ties assigned to the earlier epoch. The final model used all 50,000 labeled images, seed 42, and the weights after epoch 240. Its former holdout images therefore cannot provide an independent local test score."
    )
    p(
        "Training used cross-entropy and SGD with learning rate 0.1, momentum 0.9, weight decay 0.0005 and batch size 128. The learning rate followed cosine decay without warmup or restarts [6]. Every image appeared once per epoch. A shuffled sampler repaired missing classes by swapping examples so that each batch contained all ten classes. The final batch contained 80 images, giving 391 updates per epoch."
    )
    p(
        "The augmentations were horizontal flipping, rotation within +/-10 degrees, and Cutout with a 12 x 12 mask [3], each applied with probability 0.5. Rotation used bilinear interpolation and reflection padding. Cutout filled the selected region with normalized zero. Dropout of 0.1 was applied before each block's second convolution [4]. Per-sample residual dropping increased linearly from 0.1/9 to 0.1 across the nine blocks [5]. The model used no pretrained weights or external training data."
    )

    story.append(PageBreak())
    h("3. Experiments")
    p(
        "The base configuration used the widened network, BlurPool, Cutout and weight decay of 0.0005. I then compared all eight combinations of rotation (R), dropout (D) and residual dropping (S) at 80 epochs, together with two 240-epoch configurations. Each configuration was trained with seeds 142, 143 and 144, for 30 completed runs. The same 5,000 validation images were used throughout."
    )
    p(
        "For every run, I evaluated both raw inference and horizontal-flip test-time augmentation (TTA) at the same saved checkpoint. TTA averages the softmax probabilities of the original image and its horizontal mirror. It doubles inference computation without adding parameters. The configuration and inference policy with the largest total number of correct validation predictions across the three seeds was selected."
    )
    with (data_root / "evidence/confirmation_inference_summary.csv").open() as file:
        summary = list(csv.DictReader(file))
    names = list(dict.fromkeys(row["configuration"] for row in summary))
    data = {(row["configuration"], row["policy"]): row for row in summary}
    p(
        "Table 1. Mean validation accuracy (%) across three seeds. A 1 enables the corresponding factor. All rows use 2,436,346 parameters.",
        small,
    )
    table(
        ["R / D / S", "Epochs", "Raw", "Flip TTA"],
        [
            [
                name.split("_")[1],
                data[name, "raw"]["epochs"],
                f"{float(data[name,'raw']['mean_accuracy_percent']):.3f}",
                f"{float(data[name,'tta']['mean_accuracy_percent']):.3f}",
            ]
            for name in names
        ],
        [130, 95, 110, 110],
    )
    p(
        "The all-factor model trained for 240 epochs had the highest mean accuracy: 95.967% with raw inference and 96.040% with TTA. Its TTA accuracies were 96.06%, 95.88% and 96.18%, giving a seed standard deviation of 0.151 percentage points. At 240 epochs, adding the three regularizers increased raw mean accuracy by 0.540 points over the base. Extending the all-factor model from 80 to 240 epochs added 0.527 points, at roughly three times the training budget."
    )
    p(
        "The small average gain from TTA was less certain. Its paired gain was 0.073 points, with an unadjusted 95% t interval of [-0.396, 0.543]. None of the 34 planned comparisons passed Holm correction at 0.05. I selected the highest observed mean, but these results do not establish a reliable benefit from every individual component. The three seeds measure training variability on one split, and repeated checkpoint and configuration selection can overstate validation performance."
    )

    h("4. Final result and discussion")
    p(
        "After selection, I retrained the chosen configuration from scratch on all 50,000 labeled images and used horizontal-flip TTA for the second submission. The model reached 87.90% public accuracy, compared with 85.00% for the earlier core240/raw submission. The gain was 2.90 percentage points. Since the regularization, training-set size and inference policy changed together, this comparison cannot isolate the contribution of one change. The private leaderboard score is not recorded."
    )
    p(
        "The public score is lower than the development validation mean. Recomputing predictions from the released checkpoint reproduced all 2,000 submitted labels, so the checks did not find an export or ID-order error. Evaluation-data differences and repeated validation selection may contribute to the gap, but the hidden labels do not allow its cause to be established. I would use a fresh validation design before extending the search further."
    )

    h("5. Training record and reproducibility")
    with (data_root / "evidence/history.csv").open() as file:
        history = list(csv.DictReader(file))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.25), layout="constrained")
    epochs = [int(row["epoch"]) for row in history]
    axes[0].plot(
        epochs, [float(row["train_loss"]) for row in history], color="black", lw=1
    )
    axes[1].plot(epochs, [float(row["lr"]) for row in history], color="black", lw=1)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", ylim=(0, None), xlim=(0, 240))
    axes[1].set(xlabel="Epoch", ylabel="Learning rate", ylim=(0, None), xlim=(0, 240))
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=240)
    plt.close(fig)
    stream.seek(0)
    story.append(Image(stream, width=468, height=162))
    p(
        "Figure 1. Training loss and learning rate for the final 240-epoch fit. Loss is measured on augmented training images and is not a test metric.",
        small,
    )
    p(
        "Training was run on one NVIDIA RTX 5070 Ti using Python 3.13.5, PyTorch 2.11.0+cu128 and CUDA 12.8. Training used BF16; prediction used FP32. The complete run contains 93,840 optimizer updates. Normalization was recomputed on the full training set before this fit."
    )
    p(
        'The repository linked on the first page contains the model, runnable training and prediction code, configuration, logs and an executed review notebook. The notebook prints the public accuracy and parameter count and recomputes the validation results in Table 1. To reproduce the submission, place the course data in <font name="Courier" size="9">data/competition</font> and run <font name="Courier" size="9">python predict.py --data-dir data/competition</font>. Detailed environment setup and verification commands are in the README.'
    )

    h("References")
    references = [
        (
            "[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. <i>CVPR</i>, 2016.",
            "https://arxiv.org/abs/1512.03385",
            "arXiv:1512.03385",
        ),
        (
            "[2] R. Zhang. Making Convolutional Networks Shift-Invariant Again. <i>ICML</i>, 2019.",
            "https://arxiv.org/abs/1904.11486",
            "arXiv:1904.11486",
        ),
        (
            "[3] T. DeVries and G. W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017.",
            "https://arxiv.org/abs/1708.04552",
            "arXiv:1708.04552",
        ),
        (
            "[4] N. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. <i>JMLR</i>, 15:1929-1958, 2014.",
            "https://jmlr.org/papers/v15/srivastava14a.html",
            "JMLR 15",
        ),
        (
            "[5] G. Huang et al. Deep Networks with Stochastic Depth. <i>ECCV</i>, 2016.",
            "https://arxiv.org/abs/1603.09382",
            "arXiv:1603.09382",
        ),
        (
            "[6] I. Loshchilov and F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. <i>ICLR</i>, 2017.",
            "https://arxiv.org/abs/1608.03983",
            "arXiv:1608.03983",
        ),
        (
            "[7] NYU Tandon. <i>Deep Learning Project 1, Fall 2026.</i> Course assignment and competition rules.",
            "https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules",
            "Kaggle rules",
        ),
    ]
    for text, url, label in references:
        p(text + f' <link href="{url}" color="black">{label}</link>.', small)
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=72,
        rightMargin=72,
        topMargin=54,
        bottomMargin=54,
        title="CIFAR-10 Classification with a Modified ResNet",
        author="Zihan Xu",
    ).build(story, onFirstPage=page_number, onLaterPages=page_number)
    count = len(PdfReader(output).pages)
    assert count <= 8, count
    print(f"{output}: {count} pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "report.pdf"
    )
    args = parser.parse_args()
    build(args.data_root.resolve(), args.output.resolve())
