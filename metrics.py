import json
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14, 8)

# Чтение данных
with open('output_maskrcnn_r101/metrics.json') as f:
    data = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(data)

# 1. График потерь (Losses)
plt.figure()
plt.plot(df['iteration'], df['total_loss'], 'b-', linewidth=2, label='Total Loss')
plt.plot(df['iteration'], df['loss_cls'], 'r--', label='Classification Loss')
plt.plot(df['iteration'], df['loss_box_reg'], 'g--', label='Box Regression Loss')
plt.plot(df['iteration'], df['loss_mask'], 'm--', label='Mask Loss')
plt.plot(df['iteration'], df['loss_rpn_cls'], 'c--', label='RPN Class Loss')
plt.plot(df['iteration'], df['loss_rpn_loc'], 'y--', label='RPN Loc Loss')
plt.xlabel('Итерация', fontsize=12)
plt.ylabel('Значение потери', fontsize=12)
plt.title('Потери по итерациям', fontsize=14)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot1.png')
plt.show()

# 2. График точности (Accuracy)
plt.figure()
plt.plot(df['iteration'], df['fast_rcnn/cls_accuracy'], 'b-', label='Cls Accuracy')
plt.plot(df['iteration'], df['fast_rcnn/fg_cls_accuracy'], 'r-', label='FG Cls Accuracy')
plt.plot(df['iteration'], df['mask_rcnn/accuracy'], 'g-', label='Mask Accuracy')
plt.xlabel('Итерация', fontsize=12)
plt.ylabel('Точность', fontsize=12)
plt.title('Точность по итерациям', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plot2.png')
plt.show()

# 3. График ложных срабатываний
plt.figure()
plt.plot(df['iteration'], df['fast_rcnn/false_negative'], 'b-', label='Fast RCNN FN')
plt.plot(df['iteration'], df['mask_rcnn/false_negative'], 'r-', label='Mask RCNN FN')
plt.plot(df['iteration'], df['mask_rcnn/false_positive'], 'g-', label='Mask RCNN FP')
plt.xlabel('Итерация', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.title('Частота ложно позитивных/негативных срабатываний', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot3.png')
plt.show()

# 4. График анкеров и learning rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Анкеры
ax1.plot(df['iteration'], df['rpn/num_pos_anchors'], 'b-', label='Pos Anchors')
ax1.plot(df['iteration'], df['rpn/num_neg_anchors'], 'r-', label='Neg Anchors')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Count')
ax1.set_title('RPN Anchors Statistics')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Learning rate
ax2.plot(df['iteration'], df['lr'], 'g-', marker='o', markersize=4)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot4.png')
plt.show()