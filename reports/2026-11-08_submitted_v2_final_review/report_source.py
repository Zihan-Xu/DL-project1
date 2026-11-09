"""Rebuild the eight-page report using the supplied frozen numeric evidence.

Usage: python report_source.py --data-root . --output report.pdf
"""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
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


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validate_evidence(root):
    """Verify source hashes, run counts, and all four development summary tables."""
    dev = root / "evidence/development"
    sources = json.loads((dev / "sources.json").read_text())
    for name, metadata in sources.items():
        assert (
            hashlib.sha256((dev / name).read_bytes()).hexdigest() == metadata["sha256"]
        ), name
    specs = [
        ("architecture", "configuration", "accuracy_mean_pct", "accuracy_sd_pp", 42),
        ("data", "arm", "mean", "sd", 71),
        ("training", "config_id", "mean", "sd", 120),
        ("combination", "configuration", "mean_accuracy_percent", "sd_accuracy_pp", 86),
    ]
    for study, key, mean_key, sd_key, expected in specs:
        runs = read_rows(dev / f"{study}_runs.csv")
        summary = read_rows(dev / f"{study}_summary.csv")
        assert len(runs) == expected
        assert len({(r[key], r["seed"]) for r in runs}) == expected
        for row in summary:
            values = [
                int(r["validation_correct"]) / 50 for r in runs if r[key] == row[key]
            ]
            assert len(values) == int(row["n"]) == int(row["planned_n"])
            assert math.isclose(
                statistics.mean(values), float(row[mean_key]), abs_tol=1e-9
            )
            assert math.isclose(
                statistics.stdev(values), float(row[sd_key]), abs_tol=1e-9
            )
        audit = json.loads((dev / f"{study}_acceptance.json").read_text())
        assert audit["passed"], study
    confirmation = read_rows(root / "evidence/confirmation_inference_run_metrics.csv")
    assert len(confirmation) == 60
    for row in read_rows(root / "evidence/confirmation_inference_summary.csv"):
        values = [
            float(r["validation_accuracy"]) * 100
            for r in confirmation
            if r["configuration"] == row["configuration"]
            and r["policy"] == row["policy"]
        ]
        assert len(values) == int(row["n"]) == 3
        assert math.isclose(
            statistics.mean(values), float(row["mean_accuracy_percent"]), abs_tol=1e-9
        )
    return dev


def build(data_root, output):
    dev = validate_evidence(data_root)
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
    author = ParagraphStyle("Author", parent=body, alignment=TA_CENTER, spaceAfter=5)
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
    small = ParagraphStyle(
        "Small", parent=body, fontSize=9.5, leading=11.5, spaceAfter=6
    )
    cell = ParagraphStyle("Cell", parent=body, fontSize=10, leading=12, spaceAfter=0)
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def h(text):
        p(text, heading)

    def page():
        story.append(PageBreak())

    def table(caption, headers, rows, widths):
        p(caption, small)
        values = [[Paragraph("<b>" + str(x) + "</b>", cell) for x in headers]]
        values.extend([[Paragraph(str(x), cell) for x in row] for row in rows])
        t = Table(values, colWidths=widths, repeatRows=1, hAlign="CENTER")
        t.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
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
    h("1. Overview and data preparation")
    p(
        "My second competition submission achieved <b>87.90% accuracy on the Kaggle public test set</b> with <b>2,436,346 trainable parameters</b>. It uses one modified ResNet-20 trained from scratch on all 50,000 permitted images. Development began with a narrow baseline and progressed through width, augmentation, architecture, data-sampling and optimization studies. BlurPool, Cutout and stronger weight decay formed the core recipe; a separate confirmation study selected small rotations, dropout, residual dropping and longer training. This report follows those experiments and the decisions behind the submitted model."
    )
    p(
        "The supplied data contain 50,000 labeled and 2,000 competition images, each 32 x 32 RGB. I checked archive contents, image decoding, IDs, labels and pixel hashes. The audit found no exact duplicate images, conflicting duplicate labels or exact training/competition overlaps. A fixed stratified split, seed 42, allocated 40,000 images to training, 5,000 to validation and 5,000 to a historical local test: 4,000/500/500 per class. Development normalization used training pixels only: RGB mean (0.491882, 0.482736, 0.446956), standard deviation (0.246919, 0.243416, 0.261570)."
    )
    h("2. Baseline and model capacity")
    p(
        "The baseline followed CIFAR ResNet [1]: a 3 x 3 stem, three stages of three two-convolution residual blocks, global average pooling and a ten-class classifier. There was no initial pooling. I trained for 80 epochs with cross-entropy, SGD (learning rate 0.1, momentum 0.9, weight decay 0.0001), batch size 128 and cosine decay. The initial augmentation was reflection-padded random cropping and horizontal flipping. The epoch-75 checkpoint reached 90.94% validation and 90.66% historical local-test accuracy."
    )
    width = read_rows(dev / "width.csv")
    table(
        "Table 1. Width sweep: 80 epochs, seed 42, identical crop-and-flip recipe.",
        ["Width", "Stage channels", "Parameters", "Validation (%)"],
        [
            [
                r["width"] + "x",
                r["channels"],
                f'{int(r["parameters"]):,}',
                f'{100 * float(r["validation_accuracy"]):.2f}',
            ]
            for r in width
        ],
        [49, 139, 139, 141],
    )
    p(
        "Widening [2] to 48/96/192 channels gave the best validation result. The 4x model remained under the five-million limit but used about 78% more multiply-accumulates than 3x and did not improve accuracy in this run. I retained 3x; its historical local-test accuracy was 93.52%. This was an initial capacity choice from one seed, not evidence that 3x always outperforms 4x."
    )

    page()
    h("3. Initial augmentation and background experiments")
    p(
        "I first separated augmentation effects using the narrow model, the same split and seed 42. These arms removed random cropping and applied only the named transformation with probability 0.5; the no-augmentation control used normalization alone. Thus, Table 2 compares each transformation with no augmentation, not with the earlier crop-and-flip baseline."
    )
    aug_names = {
        "A0_none": "None",
        "A1_flip": "Horizontal flip",
        "A2_rotation": "Rotation within +/-10 degrees",
        "A3_color": "Brightness, contrast, saturation",
        "A4_resolution": "Resize 32 to 28 to 32",
        "A5_noise": "Gaussian noise, SD 0.02",
        "A6_background": "Background replacement",
    }
    aug = [r for r in read_rows(dev / "augmentation.csv") if r["arm"] in aug_names]
    table(
        "Table 2. Initial single-seed augmentation study; all runs completed 80 epochs. pp = percentage points; SD = standard deviation.",
        ["Transformation", "Validation (%)", "Change (pp)"],
        [
            [
                aug_names[r["arm"]],
                f'{float(r["validation_accuracy_pct"]):.2f}',
                f'{float(r["delta_vs_A0_pp"]):+.2f}',
            ]
            for r in aug
        ],
        [258, 108, 102],
    )
    p(
        "Flipping and small rotations helped in this setting. Rotation used bilinear interpolation with reflection padding. Color factors were sampled from 0.9 to 1.1, and noise was clipped to the valid pixel range. Color, lower resolution and noise did not improve the control. This suggested that changing geometry was more useful than adding these particular pixel perturbations; it did not rule out other strengths or combinations."
    )
    p(
        "For background replacement, I used GrabCut [3] to estimate foregrounds, removed a donor foreground by inpainting, and pasted the source foreground at its original location and size. The source label was retained. A second segmentation of the composite had to overlap the source mask with intersection-over-union of at least 0.75. This produced 7,710 accepted training pairs. All source and donor images came from the training split. A stable mask could still select the wrong object, so mask agreement alone was not sufficient quality control."
    )
    h("3.1 Paired fine-tuning")
    p(
        "I inspected 1,000 candidate pairs, sampled as 100 per class, and retained 750. Starting from the selected 3x checkpoint, I compared ten further epochs of ordinary cross-entropy with cross-entropy plus a pair-weighted supervised contrastive loss [4]. Both arms saw the 40,000 originals and 750 composites once per epoch, with each original/composite pair in the same batch. They used learning rate 0.001, cosine decay and no additional image augmentation. The contrastive term used normalized pooled features, temperature 0.1 and loss weight 0.1."
    )
    p(
        "The parent scored 93.44% on validation; cross-entropy fine-tuning scored 93.28%, and the contrastive arm scored 93.10%. Both new best checkpoints occurred after the first epoch. Although cross-entropy fine-tuning reached 93.66% on the historical local test, I kept the parent because the validation comparison did not support replacing it. Manual filtering was subjective, and the 750 examples were too small a pool to justify a general claim about background invariance."
    )

    page()
    h("4. Systematic architecture and data studies")
    p(
        "I next planned 34 research directions: eight architecture, twelve data and fourteen training directions. Each configuration used seeds 42-44; designated comparisons also used 45-46. The default was the 3x model, 80 epochs, flip probability 0.5 and no random crop. Each study kept its own matched control and normalization protocol. The sampling study shared training-only subset statistics within each seed. The different augmentation protocols prevent a controlled comparison with the earlier width sweep."
    )
    p(
        "Checkpoints maximized correct validation predictions, with ties assigned to the earlier epoch. Comparisons used common seeds only. Paired 95% t intervals measure training variability on this fixed split, not uncertainty over new datasets. Repeated predictions on the same images are not independent observations; exploratory intervals do not automatically correct for multiple comparisons."
    )
    h("4.1 Architecture: 42 completed runs")
    p(
        "The architecture study tested deeper networks at fixed width, depth/width tradeoffs near equal parameter count, early versus late channel allocation, pre-activation, two channel-attention designs (SE and ECA), grouped convolutions and BlurPool. At fixed width, ResNet-32 lost 0.79 percentage points against its matched ResNet-20 control. Equal-parameter depth changes, attention and grouping did not show a clear accuracy advantage. The grouped variant reduced theoretical computation but was slower on the measured GPU; operation count alone was a poor latency proxy."
    )
    p(
        "BlurPool [5] was the clearest candidate: 92.61% mean validation accuracy over three seeds and a paired gain of 2.47 points (interval 1.44 to 3.49). It added no trainable weights, but increased multiply-accumulates by 19.6% and measured training time by about 55% against its matched control. I retained it for the later combination study rather than assuming that its isolated gain would carry over unchanged."
    )
    h("4.2 Data: 71 verified results, including six reused runs")
    p(
        "Sixty-five data runs were newly trained. Six earlier label/tone sampling results were reused only after checking split, normalization, initialization, per-epoch sampling and validation replay. Subset experiments equalized image presentations at 40,000 per epoch. A fixed balanced 20,000-image pool averaged 85.57%, and a 10,000-image pool averaged 78.91%; paired losses against full-data training were 4.71 and 11.37 points. Rotating the 20,000-image selection recovered 4.45 points against the fixed pool. Repeatedly seeing fewer unique examples did not replace data coverage."
    )
    data = {r["arm"]: r for r in read_rows(dev / "data_summary.csv")}
    comparisons = {
        r["treatment"]: r
        for r in read_rows(dev / "data_comparisons.csv")
        if r["control"] == "full_flip"
    }
    table(
        "Table 3. Selected full-data augmentation results against the matched flip control.",
        ["Addition", "Seeds", "Mean (%)", "Paired change (pp)"],
        [
            [
                label,
                data[key]["n"],
                f'{float(data[key]["mean"]):.3f}',
                f'{float(comparisons[key]["mean"]):+.3f}',
            ]
            for key, label in [
                ("rotation10", "Rotation +/-10 degrees"),
                ("cutout12", "Cutout 12 x 12"),
                ("grayscale10", "Grayscale, probability 0.1"),
                ("background_manual750", "750 filtered backgrounds"),
            ]
        ],
        [220, 45, 82, 121],
    )
    p(
        "Cutout [6] improved all five seeds: +1.288 points, interval [0.786, 1.790]. Strictly equal class counts in every batch reduced accuracy by 0.644 points. Balancing 20,000-image subsets by color tone changed the mean by only +0.073 points; brightness balancing gave +0.887. Background quality and per-class quota variants gave no clear improvement. I kept full data and Cutout for the core recipe; the rotation result justified further testing."
    )

    page()
    h("5. Optimization and regularization experiments")
    p(
        "The fourteen training directions covered epoch budget, learning rate and warmup, learning-rate schedule, weight decay, optimizer, batch size, label smoothing, dropout, residual dropping, sharpness-aware minimization (SAM), stochastic weight averaging (SWA), exponential moving averages (EMA), supervised contrastive learning and distillation [4, 7-11]. There were 34 configurations and 120 registered outputs, all completed. These came from 107 training runs because some averaging methods shared a run; counting each output as a separate run would double-count part of the work."
    )
    p(
        "Most comparisons changed one setting in the 80-epoch, flip-only 3x recipe. Longer runs used 160 or 240 epochs; batch sizes 64 and 256 were compared with 128. I tried learning rates 0.05 and 0.2, five-epoch warmup, multistep decay and cosine restarts. Optimizer trials included Nesterov SGD and AdamW at two learning rates. The stored records include validation predictions, per-class recalls, model size, compute and timing, so reported accuracy could be checked against saved predictions."
    )
    train = {r["config_id"]: r for r in read_rows(dev / "training_summary.csv")}
    train_cmp = {r["config_id"]: r for r in read_rows(dev / "training_comparisons.csv")}
    chosen = [
        ("wd0003", "Weight decay 0.0003"),
        ("wd0005", "Weight decay 0.0005"),
        ("epochs240", "240 epochs"),
        ("dropout01", "Dropout 0.1"),
        ("depth01", "Residual drop, maximum 0.1"),
        ("sam005", "SAM, radius 0.05"),
        ("kd_t4", "2x student, distillation T = 4"),
    ]
    table(
        "Table 4. Selected training results. Changes use common seeds and each method's designated control; distillation uses the 2x student control.",
        ["Setting", "Seeds", "Mean (%)", "Change (pp)", "95% interval"],
        [
            [
                label,
                train[key]["n"],
                f'{float(train[key]["mean"]):.3f}',
                f'{float(train_cmp[key]["mean"]):+.3f}',
                f'[{float(train_cmp[key]["ci_low"]):.2f}, {float(train_cmp[key]["ci_high"]):.2f}]',
            ]
            for key, label in chosen
        ],
        [189, 38, 68, 77, 96],
    )
    p(
        "Weight decay 0.0005 was a consistent improvement: mean 92.24%, with a paired gain of 1.896 points over five seeds. Weight decay 0.0003 also improved its matched comparison. Of the 32 planned tests, these two changes and temperature-4 distillation passed Holm correction at 0.05. Distillation used a project-trained teacher, not external pretrained weights. It raised a 1,084,586-parameter student's mean from 89.464% to 91.300%; that result did not establish an advantage over the larger 3x model."
    )
    p(
        "Longer training and modest dropout or residual dropping had promising means, but their gains required confirmation after combining them with BlurPool and Cutout. SAM at radius 0.05 added only 0.284 points with an interval spanning zero, while its recorded mean training time rose from 4.82 to 10.65 minutes. SWA changed its matched control by -0.052 points, and EMA at decay 0.999 changed the base by +0.196; neither interval excluded zero. Label smoothing at 0.1 also lacked evidence of a benefit. I therefore kept ordinary SGD and did not add these methods to the final recipe."
    )
    p(
        "The completed training study comprised 9,280 epochs and 2,942,080 optimizer updates. Its audit checked planned seeds, checkpoint hashes, summary statistics and paired schedules. This mattered after interrupted jobs: a recovered checkpoint or a partial history was not counted as a completed result."
    )

    # Remaining pages are added below.
    build_remaining(story, p, h, table, page, small, data_root, dev)
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
    reader = PdfReader(output)
    count = len(reader.pages)
    print(f"{output}: {count} pages")
    for index, pdf_page in enumerate(reader.pages, 1):
        text = pdf_page.extract_text()
        print(
            f"Page {index}: {len(text.split())} words; starts {text.splitlines()[1:3]}"
        )
    assert count == 8, f"Expected exactly eight reviewed pages, got {count}"


def build_remaining(story, p, h, table, page, small, data_root, dev):
    page()
    h("6. Combining the promising components")
    p(
        "The next study tested combinations directly. I completed 86 training runs across 22 configurations, rather than adding gains from separate studies. The central experiment was a full factorial over BlurPool (B), Cutout 12 (C), and weight decay 0.0005 instead of 0.0001 (W). All eight combinations used 80 epochs and five seeds, 42-46, with the same split and horizontal-flip augmentation."
    )
    combinations = {
        r["configuration"]: r for r in read_rows(dev / "combination_summary.csv")
    }
    table(
        "Table 5. Core factorial: five-seed mean and sample SD on the shared validation set. A 1 enables the factor; a 0 disables it.",
        ["B / C / W", "Mean accuracy (%)", "Seed SD (pp)"],
        [
            [
                name[-3:],
                f'{float(combinations[name]["mean_accuracy_percent"]):.3f}',
                f'{float(combinations[name]["sd_accuracy_pp"]):.3f}',
            ]
            for name in combinations
            if name.startswith("core_") and name[5:].isdigit()
        ],
        [133, 189, 146],
    )
    p(
        "The complete core reached 94.648%, compared with 90.340% for its matched base. The paired gain was 4.308 points, with an interval of [3.807, 4.809] and Holm-adjusted p = 0.00060. This directly supported the combined recipe. It did not imply that the separate improvements were additive: the factorial allowed the components to change one another's effect."
    )
    p(
        "Extensions of this core tested smaller Cutout, weaker weight decay, rotation, dropout, residual dropping, attention, a smaller batch, warmup, two training views and 240 epochs. The three-seed means were 95.167% for +/-10-degree rotation, 95.200% for dropout 0.1 and 95.107% for maximum residual drop 0.1. The 240-epoch core reached 95.387%, at approximately three times the training cost. These were candidates for a fresh joint comparison, not proof that combining all of them would win."
    )
    p(
        "The distillation gain also changed after strengthening the recipe. A 2x student averaged 94.096% with ordinary cross-entropy and 94.204% with temperature-4 distillation over five seeds. The paired gain was only 0.108 points, interval [-0.131, 0.347], while measured training cost was about 1.93 times higher. I did not use distillation for the submitted 3x model. All 86 checkpoints were independently replayed in FP32, reproducing 430,000 validation predictions and their saved logits."
    )
    h("6.1 Testing inference transformations")
    p(
        "On the three 240-epoch core models, averaging the original and horizontally flipped image probabilities raised mean accuracy from 95.387% to 95.767%. Adding +/-5-degree rotated views with their mirrors gave 95.147%, and +/-10-degree views with mirrors gave 95.367%; neither beat flip-only inference. A separate three-seed training test with right-angle rotations averaged 93.82%, versus 94.57% for its matched 80-epoch flip-only control. I retained modest training rotations for confirmation and kept inference to original plus mirror."
    )

    page()
    h("7. Final confirmation and model selection")
    p(
        "I then used new training seeds 142, 143 and 144 for a 30-run confirmation study. The core always included the 3x network, BlurPool, Cutout 12 and weight decay 0.0005. At 80 epochs I tested all eight combinations of rotation (R), dropout (D) and residual dropping (S). At 240 epochs I compared the core alone with all three additions. Every planned run finished before selecting the final configuration."
    )
    p(
        "For each run, raw and horizontal-flip test-time augmentation (TTA) used the same checkpoint, selected by raw validation accuracy. TTA averaged softmax probabilities, not class labels or logits. I selected the configuration/policy with the largest total correct count across the three seeds; the recorded tie rule preferred raw inference, then fewer epochs, then configuration name. This used all seeds rather than the single best run."
    )
    summary = read_rows(data_root / "evidence/confirmation_inference_summary.csv")
    names = list(dict.fromkeys(r["configuration"] for r in summary))
    lookup = {(r["configuration"], r["policy"]): r for r in summary}
    table(
        "Table 6. Complete confirmation study: three-seed mean validation accuracy (%). R = +/-10-degree rotation; D = dropout 0.1; S = maximum residual drop 0.1.",
        ["R / D / S", "Epochs", "Raw", "Flip TTA"],
        [
            [
                name.split("_")[1],
                lookup[name, "raw"]["epochs"],
                f'{float(lookup[name, "raw"]["mean_accuracy_percent"]):.3f}',
                f'{float(lookup[name, "tta"]["mean_accuracy_percent"]):.3f}',
            ]
            for name in names
        ],
        [135, 93, 120, 120],
    )
    p(
        "The selected configuration, long_111 with TTA, averaged 96.040%. Its individual accuracies were 96.06%, 95.88% and 96.18%, giving a sample standard deviation of 0.151 percentage points. Raw inference averaged 95.967%. At 240 epochs the three additions improved raw mean accuracy by 0.540 points over the core, while extending the all-factor model from 80 to 240 epochs improved it by 0.527 points."
    )
    p(
        "The final TTA gain was small: +0.073 points, with a paired interval of [-0.396, 0.543]. None of the 34 planned comparisons in this confirmation study passed Holm correction at 0.05. I chose the highest observed mean, but cannot claim a statistically established benefit for each addition. The new seeds test sensitivity to training randomness; they do not make the reused validation images a new independent test set."
    )
    p(
        "The audit verified all 30 runs and 60 raw/TTA prediction tables. The final full-data fit used seed 42 as specified in the release plan, rather than selecting whichever confirmation seed had the highest accuracy. No later model or multi-checkpoint ensemble was substituted into the second submission."
    )

    page()
    h("8. Submitted architecture and full-data training")
    p(
        "The final network has stage widths 48/96/192 and nine residual blocks. At the two stage transitions, a fixed 3 x 3 binomial filter, the outer product of [1, 2, 1] with itself, divided by 16, downsamples both paths with reflection padding. A learned 1 x 1 projection matches shortcut channels. Dropout is applied before each block's second convolution. Per-example residual dropping increases linearly from 0.1/9 to 0.1; surviving residuals are rescaled during training. The residual branch is computed before masking, so this implementation does not save training computation by skipping blocks."
    )
    table(
        "Table 7. Exact recipe for the second submitted checkpoint.",
        ["Item", "Setting"],
        [
            ["Fit", "50,000 images; seed 42; 240 epochs; final epoch weights"],
            ["Optimizer / loss", "SGD; learning rate 0.1; momentum 0.9; cross-entropy"],
            [
                "Schedule / decay",
                "Cosine to zero; no warmup or restarts; weight decay 0.0005",
            ],
            [
                "Batch / coverage",
                "128; final batch 80; each training image once per epoch",
            ],
            [
                "Augmentation",
                "Flip, rotation +/-10 degrees, Cutout 12; each probability 0.5",
            ],
            ["Regularization", "Dropout 0.1; maximum residual drop 0.1"],
            [
                "Precision / inference",
                "BF16 training; FP32 inference; original + horizontal mirror",
            ],
        ],
        [116, 352],
    )
    p(
        "Rotation uses bilinear interpolation and reflection padding. Cutout fills normalized pixels with zero and clips the mask at image boundaries. There is no random crop or Mixup in this version. A shuffled sampler swaps examples to ensure each batch contains all ten classes without duplicating or dropping images; it does not force equal class counts. Training therefore has 391 updates per epoch, 93,840 updates in total and 12 million image presentations."
    )
    p(
        "After selection, normalization was recomputed on all 50,000 images: mean (0.491400, 0.482158, 0.446531), standard deviation (0.247032, 0.243485, 0.261588). The fit used no validation-based early stopping. No external data or pretrained weights were used, and 2,436,346 parameters are below the course limit of five million [13]. Training ran on one NVIDIA RTX 5070 Ti with Python 3.13.5, PyTorch 2.11.0+cu128 and CUDA 12.8."
    )
    history = read_rows(data_root / "evidence/history.csv")
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
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.0), layout="constrained")
    epochs = [int(r["epoch"]) for r in history]
    axes[0].plot(epochs, [float(r["train_loss"]) for r in history], color="black", lw=1)
    axes[1].plot(epochs, [float(r["lr"]) for r in history], color="black", lw=1)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", ylim=(0, None), xlim=(0, 240))
    axes[1].set(xlabel="Epoch", ylabel="Learning rate", ylim=(0, None), xlim=(0, 240))
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=240)
    plt.close(fig)
    stream.seek(0)
    story.append(Image(stream, width=468, height=144))
    p(
        "Figure 1. Recorded loss on augmented training images and cosine learning rate [12] for the final fit. Training loss is not an independent evaluation metric.",
        small,
    )

    page()
    h("9. Competition result, limitations and reproducibility")
    p(
        "The second submission achieved 87.90% accuracy on the Kaggle public test set, compared with 85.00% for the earlier core240/raw submission: a gain of 2.90 percentage points. The regularization settings, training-set size and inference policy changed together, so this is not a controlled estimate of one method's effect. The private score is not recorded. The 96.04% validation mean belongs to three 40k-trained research models, not the final 50k model, which has no independent local holdout."
    )
    p(
        "The validation/public gap remains unexplained. Repeated selection on one validation split may make its best result optimistic, and the evaluation datasets may differ; hidden labels prevent testing these explanations. The released checkpoint reproduced all 2,000 submitted labels, and the downloaded submitted file matched byte for byte. The replay found no mismatch in the exported labels or their order; it cannot explain the lower public score."
    )
    p(
        "The repository includes the checkpoint, exact configuration, training/evaluation/inference code, logs, source tables and an executed review notebook. Development training scripts are supplied in the research directory. With course data installed as described in the README, run <font name='Courier' size='9'>python predict.py --data-dir data/competition</font> and <font name='Courier' size='9'>python verify.py</font>. The clean code matched the original initialization, forward/backward results, augmentation and first full training epoch; CPU replay matched both raw and TTA predictions. Cleanup did not repeat the full 240-epoch fit. TTA uses two passes, about 872.2 million multiply-accumulates per image, without extra parameters. The main lesson was to verify promising combinations under matched conditions and retain simpler methods when more expensive additions gave uncertain gains."
    )
    h("References")
    references = [
        (
            "[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.",
            "https://arxiv.org/abs/1512.03385",
            "arXiv:1512.03385",
        ),
        (
            "[2] S. Zagoruyko and N. Komodakis. Wide Residual Networks. BMVC, 2016.",
            "https://arxiv.org/abs/1605.07146",
            "arXiv:1605.07146",
        ),
        (
            "[3] C. Rother, V. Kolmogorov, and A. Blake. GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts. ACM TOG, 23(3), 2004.",
            "https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/",
            "Paper",
        ),
        (
            "[4] P. Khosla et al. Supervised Contrastive Learning. NeurIPS, 2020.",
            "https://arxiv.org/abs/2004.11362",
            "arXiv:2004.11362",
        ),
        (
            "[5] R. Zhang. Making Convolutional Networks Shift-Invariant Again. ICML, 2019.",
            "https://arxiv.org/abs/1904.11486",
            "arXiv:1904.11486",
        ),
        (
            "[6] T. DeVries and G. W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017.",
            "https://arxiv.org/abs/1708.04552",
            "arXiv:1708.04552",
        ),
        (
            "[7] N. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15:1929-1958, 2014.",
            "https://jmlr.org/papers/v15/srivastava14a.html",
            "JMLR 15",
        ),
        (
            "[8] G. Huang et al. Deep Networks with Stochastic Depth. ECCV, 2016.",
            "https://arxiv.org/abs/1603.09382",
            "arXiv:1603.09382",
        ),
        (
            "[9] P. Foret et al. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR, 2021.",
            "https://arxiv.org/abs/2010.01412",
            "arXiv:2010.01412",
        ),
        (
            "[10] P. Izmailov et al. Averaging Weights Leads to Wider Optima and Better Generalization. UAI, 2018.",
            "https://arxiv.org/abs/1803.05407",
            "arXiv:1803.05407",
        ),
        (
            "[11] G. Hinton, O. Vinyals, and J. Dean. Distilling the Knowledge in a Neural Network. 2015.",
            "https://arxiv.org/abs/1503.02531",
            "arXiv:1503.02531",
        ),
        (
            "[12] I. Loshchilov and F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR, 2017. The submitted recipe uses cosine decay without restarts.",
            "https://arxiv.org/abs/1608.03983",
            "arXiv:1608.03983",
        ),
        (
            "[13] NYU Tandon. Deep Learning Project 1, Fall 2026. Course assignment and competition rules.",
            "https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules",
            "Kaggle rules",
        ),
    ]
    for text, url, label in references:
        p(text + f' <link href="{url}" color="black">{label}</link>.', small)


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
