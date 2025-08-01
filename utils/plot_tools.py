# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from typing import Optional   # Python 3.6+

# def plot_two_top1_lifetime(top1_a,
#                            top1_b,
#                            bucket_hours: int = 3,
#                            lifetime_max: int = 36,
#                            save_path: Optional[str] = None):
#     """
#     두 개의 top1_lifetimes 배열을 한 그래프에 그립니다.
#     • 첫째: 파란색 ‘TUDOR’
#     • 둘째: 빨간색 ‘TUDOR w/o adjust score’
#     • 제목: Impact of adjusting score on remaining lifetime
#     • x축 major ticks: 0,3,6,…,36
#     • y축: Recommendations (%) 
#     • 히스토그램 해상도: 0.5h bins → centers에 점 표시
#     • 음수 → x=0, >36h → x=36
#     """
#     arrays = [np.asarray(top1_a, float), np.asarray(top1_b, float)]
#     labels = ['TUDOR', 'TUDOR w/o adjust score']
#     colors = ['blue', 'red']
#     totals = [arr.size for arr in arrays]

#     # 0.5h 해상도 bins, centers
#     bins = np.arange(0, lifetime_max + 0.5, 0.5)  # 0,0.5,…,36
#     centers = bins[:-1] + 0.25                   # 0.25,0.75,…,35.75

#     plt.figure(figsize=(6, 3.4), dpi=120)
#     for arr, label, color, total in zip(arrays, labels, colors, totals):
#         # 음수·초과 비율
#         neg_pct = (arr < 0).sum() / total * 100
#         ov_pct  = (arr > lifetime_max).sum() / total * 100

#         # 0~36h 구간 히스토그램
#         in_range = arr[(arr >= 0) & (arr <= lifetime_max)]
#         hist, _  = np.histogram(in_range, bins=bins)
#         pct      = hist / total * 100

#         # 중간 구간 점
#         plt.scatter(centers, pct, c=color, marker='x', label=label)
#         # 음수→0h, 초과→36h
#         plt.scatter([0],   [neg_pct], c=color, marker='x')
#         plt.scatter([36],  [ov_pct],  c=color, marker='x')

#     # x축 major ticks & limits
#     major_ticks = np.arange(0, lifetime_max + 1, bucket_hours)  # 0,3,…,36
#     plt.xticks(major_ticks)
#     plt.xlim(0, lifetime_max)
#     plt.ylim(0, None)

#     plt.xlabel("Remaining lifetime of the news (hour)")
#     plt.ylabel("Recommendations (%)")
#     plt.title("Impact of adjusting score on remaining lifetime")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.legend(loc='upper left')

#     if save_path:
#         out = Path(save_path)
#         out.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(out, bbox_inches='tight')
#         print(f"[Plot] 저장 완료 → {out}")
#     else:
#         plt.show()


# if __name__ == "__main__":
#     top1   = np.load('psj/Adressa_1w/results/lifetime_ablation/top1_epoch10_seed64.npy')
#     top1_no = np.load('psj/Adressa_1w/results/lifetime_ablation/top1_epoch _wo_as.npy')
#     plot_two_top1_lifetime(top1, top1_no,
#                            save_path=None)



import numpy as np

def save_top1_distribution(top1_path: str,
                           out_txt: str,
                           lifetime_max: float = 36.0,
                           step: float = 0.5):
    """
    top1_lifetimes 파일(.npy)을 로드하여
    - 음수 → 0, 초과 → lifetime_max 로 클립
    - (-step/2, +step/2] 구간별로 histogram을 계산하여
      0.0, 0.5, …, lifetime_max (총 73개) Hour 값과
      전체 대비 비율(Ratio)을 txt로 저장
    """
    arr = np.load(top1_path)
    total = arr.size

    # 음수→0, 초과→lifetime_max
    arr = np.clip(arr, 0.0, lifetime_max)

    # bin edges: -step/2, +step/2, …, lifetime_max+step/2
    edges = np.arange(-step/2, lifetime_max + step/2 + 1e-8, step)
    hist, _ = np.histogram(arr, bins=edges)

    # Hour 값: edge + step/2 → [0.0, 0.5, …, lifetime_max]
    hours = edges[:-1] + step/2

    # 전체 대비 비율 (%)
    ratio = hist / total * 100.0

    # 파일로 저장
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('Hour\tRatio\n')
        for h, r in zip(hours, ratio):
            f.write(f'{h:.1f}\t{r}\n')

    print(f"Saved distribution → {out_txt}")

if __name__ == '__main__':
    save_top1_distribution(
        top1_path='psj/Adressa_1w/results/lifetime_ablation/seed64/top1_epoch7_seed64_wo_as.npy',
        out_txt='psj/Adressa_1w/results/lifetime_ablation/seed64/top1_epoch7_seed64_wo_as_dist.txt'
    )
