"""Render the final course report with explicit research/deployment distinction."""
import csv,json,shutil
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from pypdf import PdfReader
TASK=Path(__file__).resolve().parents[1]
OUT=TASK/'report_rebuilt.pdf'
assert not OUT.exists(), 'Refusing to overwrite an existing rebuilt report'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=colors.HexColor('#173A61'),spaceAfter=14))
styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=9.4,leading=13.4,spaceAfter=8))
styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8.1,leading=11,spaceAfter=6))
styles['Heading2'].textColor=colors.HexColor('#173A61');story=[]
def p(t,s='Body2'):story.append(Paragraph(t,styles[s]))
def h(t):p(t,'Heading2')
def table(headers,values,widths):
    t=Table([[Paragraph(escape(str(x)),styles['Small2']) for x in r] for r in [headers]+values],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EFF6')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#CBD5E1'))]));story.extend([t,Spacer(1,9)])
def figure(n,w,h):story.append(Image(str(TASK/'figures'/n),width=w,height=h))
def page(c,d):
    c.saveState();c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#606D7A'))
    c.drawString(48,30,'Zihan Xu | Deep Learning Project 1 | long111 full50k v2 | 2026-11-06')
    c.drawRightString(A4[0]-48,30,str(d.page));c.restoreState()

p('CIFAR-10 Classification<br/>with a Regularized Residual Network','Title2')
p('<b>Deep Learning Project 1 - Fall 2026<br/>Author and Kaggle team: Zihan Xu</b>')
p('Public code, weights and executed notebook:<br/><link href="https://github.com/Zihan-Xu/DL-project1" color="#225EA8">https://github.com/Zihan-Xu/DL-project1</link>')
h('1. Overview and findings')
p('This project trains a modified residual CNN from scratch on the permitted course data, under a five-million-parameter limit. A completed confirmation study compared three regularization factors at two training budgets. Its prescribed selection rule chose a widened ResNet-20 with BlurPool, Cutout, rotation, dropout and stochastic depth, followed by horizontal-flip test-time augmentation (TTA). The selected recipe was then fitted once on all 50,000 labeled images.')
table(['Measure','Result and interpretation'],[
 ['Trainable parameters','2,436,346, including the classifier; below 5,000,000'],
 ['Research validation','96.0400% mean; SD 0.1510 percentage points; seeds 142/143/144 on the same 5,000 images'],
 ['Final deployment fit','50,000 labeled images; seed42; 240 epochs; last-epoch checkpoint'],
 ['Final independent test accuracy','Not yet available. No independent local holdout remains after the full50k fit. New Kaggle score has not been reported.'],
 ['Previous Kaggle result','85.00%, user-reported for the earlier core_ep240/raw submission; remote uploaded bytes not independently verified.'],
], [137,350])
p('The 96.04% figure belongs to the 40k-trained research models, not the final full50k model. The new submission is a reasoned deployment choice; no leaderboard improvement is claimed before official scoring.')
h('2. Architecture and design choices')
p('The network uses a 3x widened CIFAR ResNet-20: channels 48/96/192, three two-convolution residual blocks in each stage, 3x3 convolutions, batch normalization and ReLU. Stage transitions use learned 1x1 shortcut projections. Global average pooling feeds a ten-class linear head. The residual path and projection use fixed 3x3 binomial BlurPool filters before stride-two sampling. Residual connections support optimization [1]; anti-aliasing adds spatial filtering at modest extra compute [2].')
p('Dropout probability 0.1 is applied before each block\'s second convolution [4]. Per-sample stochastic-depth probability increases linearly from 0.1/9 to 0.1 across the nine blocks [5]. These regularizers can improve generalization but add stochastic optimization difficulty. Widening supplies capacity within the parameter cap, at higher computation than the narrow baseline. No attention, pretrained weights, external data or teacher is used. Estimated inference cost is 436.1 million MACs for raw inference and 872.2 million for two-view TTA.')
story.append(PageBreak())
h('3. Research protocol and comparative results')
p('The original 50,000 labeled images were split by class into 40,000 training, 5,000 validation and 5,000 historical local-test examples. Research normalization used only the 40k training split. A previous accepted combination study identified BlurPool + Cutout12 + weight decay 0.0005 as a useful base. The confirmation matrix tested all eight combinations of rotation (R), dropout (D) and stochastic depth (S) at 80 epochs, plus base-only and all-three-factor versions at 240 epochs. Each configuration used new seeds142-144: 30 complete runs in total.')
p('For every research run, the best raw-validation epoch was fixed first, with earliest-epoch tie breaking. Then raw and horizontal-flip probability averaging were evaluated at that same checkpoint. Selection maximized total validation correct across three seeds over 20 configuration/policy choices; ties favored raw, fewer epochs, then configuration name. No favorable deployment seed was selected.')
with (TASK/'evidence/confirmation_inference_summary.csv').open() as f:summary=list(csv.DictReader(f))
data={r['configuration']:{q['policy']:q for q in summary if q['configuration']==r['configuration']} for r in summary}
table(['Configuration','Epochs','Raw mean %','TTA mean %'],[[n,data[n]['raw']['epochs'],f"{float(data[n]['raw']['mean_accuracy_percent']):.4f}",f"{float(data[n]['tta']['mean_accuracy_percent']):.4f}"] for n in data],[158,65,132,132])
p('Naming: confirm_RDS uses 80 epochs; bits indicate rotation +/-10 degrees, dropout0.1 and maximum stochastic depth0.1. long_000 and long_111 use 240 epochs. All rows have three seeds and 2,436,346 parameters. Full per-seed results and all negative comparisons are included in the repository.', 'Small2')
p('At 240 epochs, all three regularizers improve the raw mean over the base by 0.5400 percentage points. At 80 epochs the improvement is 0.6133 points. These matched-budget comparisons suggest the combination is useful, but do not establish a universal or statistically certain gain. Comparing the all-factor 80-epoch and 240-epoch models gives +0.5267 points for raw inference at roughly three times the training budget.')
p('TTA adds only 0.0733 points to long_111. Its paired unadjusted 95% t interval is [-0.3962, 0.5429] points: two seeds improve and one declines. None of 34 prespecified comparisons passes the joint Holm correction at 0.05. The same validation images were repeatedly used for design and checkpoint selection; three seeds describe training randomness, not three independent datasets.')
story.append(PageBreak())
h('4. Selection, full50k training and inference')
figure('confirmation.png',480,288)
p('Figure 1. Research validation means with seed-standard-deviation error bars (not confidence intervals), on a focused accuracy scale. All30 runs and their raw/TTA results passed independent acceptance and checkpoint replay. Ranking does not prove general superiority.', 'Small2')
p('After research selection, the final model was initialized from scratch with fixed seed42 and fitted on all50k images. This changes the data protocol: the former validation and historical test examples now belong to training. RGB mean was [0.49139968, 0.48215841, 0.44653091]; standard deviation was [0.24703223, 0.24348513, 0.26158784], computed over all50k training images only.')
table(['Setting','Final recipe'],[
 ['Optimizer / loss','SGD, initial LR0.1, momentum0.9, weight decay0.0005; cross-entropy'],
 ['Budget / selection','Batch128; cosine240 epochs; no warmup; final epoch240, no early stopping'],
 ['Augmentation','Flip p0.5; uniform +/-10-degree rotation p0.5 with bilinear/reflection sampling; Cutout12 p0.5, filled with normalized zero; no random crop'],
 ['Numeric / sampling','BF16 training, FP32 inference, TF32 disabled, deterministic algorithms. Every epoch contains all50k exactly once; last batch80.'],
 ['Training totals','391 updates per epoch; 93,840 updates and 12,000,000 image presentations overall'],
 ['TTA','Single checkpoint; original + horizontal mirror; equal-weight softmax probability average, then argmax'],
], [120,367])
p('TTA doubles forward-pass cost without adding trainable parameters. No rotation/crop search, ensemble of models, hidden test labels or manual answer corrections are used. The competitor test images are read only to produce predictions after the recipe and weights are fixed.')
story.append(PageBreak())
h('5. Final fit, reproducibility and limitations')
figure('training.png',480,150)
p('Figure 2. Completed full50k training loss under augmentation and the prescribed learning-rate schedule. Training accuracy is not an independent test estimate. Every epoch and optimizer update is recorded in history.csv.', 'Small2')
p('The release contains the final weights, exact architecture/operators, configuration and environment, all240 training log rows, original source snapshot, standalone training adapter, CPU/CUDA prediction entry point, evaluation utility, executed review notebook, all30 research logs and per-image research predictions. No raw course images, credentials or unrelated files are published. The standalone adapter uses the executed training loop but replaces project-specific queue/data discovery. Full240-epoch retraining was not repeated during packaging.')
p('Verification covers all50k original image bytes and normalization; all240 sampler schedules; the parameter count; all60 research prediction tables and20 means; every submission\'s two columns,2,000 unique sample-ordered IDs and integer labels0-9; and independent CPU replay of the exported model. The report and notebook figures are rendered for inspection. Hash manifests identify immutable files. Exact reproduction can depend on the GPU and library versions; the recorded environment is Python3.13.5, PyTorch2.11.0+cu128, NumPy2.1.3, Pillow11.1.0 and an RTX5070Ti.')
table(['CSV choice','Role'],[
 ['01 full50k TTA','Recommended: selected research policy, new full-data fit'],
 ['02 full50k raw','Same weights, one-pass alternative; differs on69/2,000 predictions'],
 ['03 previous core240 flip','Previous40k-trained seed42 model with flip averaging'],
 ['04 previous core240 raw','Reference version associated with user-reported0.85000'],
], [160,327])
p('The earlier local-to-Kaggle gap cannot be explained from hidden labels. Pipeline replay checks passed; distribution difficulty is a hypothesis, not a demonstrated cause. Repeated holdout selection, small seed count, the full-data refit and an unknown competition score limit claims. The practical conclusion is to submit the fixed full50k/TTA version first and record its actual official score separately. The course report is limited to four pages; supporting evidence is retained in the public repository.')
h('References')
for t in [
 '[1] He et al. Deep Residual Learning for Image Recognition. CVPR2016. arXiv:1512.03385.',
 '[2] Zhang. Making Convolutional Networks Shift-Invariant Again. ICML2019. arXiv:1904.11486.',
 '[3] DeVries and Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017. arXiv:1708.04552.',
 '[4] Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR15,2014,1929-1958.',
 '[5] Huang et al. Deep Networks with Stochastic Depth. ECCV2016. arXiv:1603.09382.',
 '[6] Loshchilov and Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR2017. arXiv:1608.03983. This implementation uses cosine decay without restarts.',
 '[7] Course-provided Deep Learning Project1, Fall2026; Kaggle CIFAR-10 ResNet Architecture Challenge rules. Implementation is original local project code inspired by the cited methods.'
]:p(t,'Small2')
OUT.parent.mkdir(exist_ok=True,parents=True)
SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=48,title='CIFAR-10 Full50k Residual Network - Zihan Xu',author='Zihan Xu').build(story,onFirstPage=page,onLaterPages=page)
pages=len(PdfReader(OUT).pages);assert pages<=8
print(json.dumps(dict(report=str(OUT),pages=pages)))
