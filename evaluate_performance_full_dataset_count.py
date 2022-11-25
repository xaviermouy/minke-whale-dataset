# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:34:38 2022

@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#% matplotlib qt


def plot_annot_boxes(
    annot_list,
    line_width=2,
    colors=["blue"],
    labels=["Annotations"],
    title="",
):
    fig, ax = plt.subplots()
    facecolor = "none"
    alpha = 1
    for idx, annot in enumerate(annot_list):
        patches = []
        for index, row in annot.data.iterrows():
            # plot annotation boxes in Time-frquency
            x = row["time_min_offset"]
            y = row["frequency_min"]
            width = row["duration"]
            height = row["frequency_max"] - row["frequency_min"]
            rect = Rectangle(
                (x, y),
                width,
                height,
                linewidth=line_width,
                edgecolor=colors[idx],
                facecolor=facecolor,
                alpha=alpha,
                label=labels[idx],
            )
            patches.append(rect)
        p = PatchCollection(
            patches,
            edgecolor=colors[idx],
            # label=labels[idx],
            facecolor=facecolor,
        )
        ax.add_collection(p)
        # ax.add_patch(rect)
    ax.set_xlabel("Times (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid()
    ax.plot([1], [1])
    plt.show()
    ax.legend()


annot_file = r"C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\Annotations_dataset_FRA-NEFSC-CARIBBEAN-201612-MTQ annotations.nc"
detec_file = r"C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ\detections.sqlite"
out_dir = (
    r"C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results"
)
# thresholds = np.arange(0.5, 1.01, 0.01)
thresholds = np.arange(0, 1.05, 0.05)
# thresholds = [1]

target_class = "MW"
files_to_use = "both"  # 'detec', 'annot', 'both', list
date_min = "2016-12-26 04:15:00"
date_max = "2017-02-23 18:16:30"

freq_ovp = True
dur_factor_max = None
dur_factor_min = None
ovlp_ratio_min = None
remove_duplicates = False
inherit_metadata = False
filter_deploymentID = True
do_plot = False
F_beta = 1
## ############################################################################

# load ground truth data
annot = Annotation()
annot.from_netcdf(annot_file)

# load destections
detec = Annotation()
detec.from_sqlite(detec_file)

# filter dates (if dataset partially annotated)
if date_min:
    annot.filter("time_min_date >= '" + date_min + "'", inplace=True)
    detec.filter("time_min_date >= '" + date_min + "'", inplace=True)
if date_max:
    annot.filter("time_max_date <= '" + date_max + "'", inplace=True)
    detec.filter("time_max_date <= '" + date_max + "'", inplace=True)

# filter to the class of interest
annot.filter('label_class == "' + target_class + '"', inplace=True)
detec.filter('label_class == "' + target_class + '"', inplace=True)

# Define list of files to use for the performance evaluation.
if files_to_use == "detec":  # case 1: all files with detections are used
    files_list = list(set(detec.data.audio_file_name))
elif files_to_use == "annot":  # case 2: all files with annotations are used
    files_list = list(set(annot.data.audio_file_name))
elif (
    files_to_use == "both"
):  # case 3: all files with annotations or detections are used
    files_list1 = list(set(annot.data.audio_file_name))
    files_list2 = list(set(detec.data.audio_file_name))
    files_list = list(set(files_list1 + files_list2))

elif (
    type(files_to_use) is list
):  # case 4: only files provided by the user are used
    files_list = files_to_use
files_list.sort()


# filter annotations with selected files to use
annot.filter(
    "audio_file_name in @files_list", files_list=files_list, inplace=True
)

# filter detections with selected files to use
detec.filter(
    "audio_file_name in @files_list", files_list=files_list, inplace=True
)

# loop through thresholds
# for th_idx, threshold in enumerate(
#     tqdm(thresholds, desc="Progress", leave=True, miniters=1, colour="green")
# ):
for th_idx, threshold in enumerate(thresholds):
    print("Threshold value: ", threshold)
    # filter detections for that threshold value
    detec_conf = detec.filter("confidence >= " + str(threshold), inplace=False)
    # init
    FP = np.zeros(len(files_list))
    TP = np.zeros(len(files_list))
    FN = np.zeros(len(files_list))
    # loop through each file
    # for idx, file in enumerate(
    #     tqdm(files_list, desc="File", leave=True, miniters=1, colour="red")
    # ):
    # for idx, file in enumerate(files_list):
    for idx, file in enumerate(
        tqdm(
            files_list,
            desc="Progress",
            leave=True,
            miniters=1,
            colour="green",
        )
    ):
        # print(idx)
        # filter to only keep data from this file
        annot_tmp = annot.filter("audio_file_name == '" + file + "'")
        detec_tmp = detec_conf.filter(
            "audio_file_name=='" + file + "'", inplace=False
        )

        # count FP, TP, FN:
        if len(annot_tmp) == 0:  # if no annotations -> FP = nb of detections
            FP[idx] = len(detec_tmp)
        elif len(detec_tmp) == 0:  # if no detections -> FN = nb of annotations
            FN[idx] = len(annot_tmp)
        else:
            ovlp = annot_tmp.filter_overlap_with(
                detec_tmp,
                freq_ovp=freq_ovp,
                dur_factor_max=dur_factor_max,
                dur_factor_min=dur_factor_min,
                ovlp_ratio_min=ovlp_ratio_min,
                remove_duplicates=remove_duplicates,
                inherit_metadata=inherit_metadata,
                filter_deploymentID=filter_deploymentID,
                inplace=False,
            )
            FN[idx] = len(annot_tmp) - len(ovlp)
            TP[idx] = len(ovlp)
            FP[idx] = len(detec_tmp) - len(ovlp)

        # Sanity check
        if FP[idx] + TP[idx] != len(detec_tmp):
            raise Exception(
                "FP and TP don't add up to the total number of detections"
            )
        elif TP[idx] + FN[idx] != len(annot_tmp):
            raise Exception(
                "FP and FN don't add up to the total number of annotations"
            )

        # plot annot and detec boxes
        if do_plot:
            plot_annot_boxes(
                [annot_tmp, detec_tmp],
                line_width=2,
                colors=["blue", "red"],
                labels=["Annotations", "Detections"],
                title=file
                + "\n TP:"
                + str(TP[idx])
                + " - FP: "
                + str(FP[idx])
                + " - FN: "
                + str(FN[idx]),
            )
    # create df for each file with threshold, TP, FP, FN
    tmp = pd.DataFrame(
        {
            "file": files_list,
            "threshold": threshold,
            "TP": TP,
            "FP": FP,
            "FN": FN,
        }
    )
    if th_idx == 0:
        performance_per_file_count = tmp
    else:
        performance_per_file_count = pd.concat(
            [performance_per_file_count, tmp], ignore_index=True
        )

    # Calculate P, R, and F for that confidence threshold
    perf_tmp = performance_per_file_count.query(
        "threshold ==" + str(threshold)
    )
    TP_th = perf_tmp["TP"].sum()
    FP_th = perf_tmp["FP"].sum()
    FN_th = perf_tmp["FN"].sum()
    R_th = TP_th / (TP_th + FN_th)
    P_th = TP_th / (TP_th + FP_th)
    F_th = (1 + F_beta**2) * ((P_th * R_th) / ((F_beta**2 * P_th) + R_th))
    tmp_PRF = pd.DataFrame(
        {
            "threshold": [threshold],
            "TP": [TP_th],
            "FP": [FP_th],
            "FN": [FN_th],
            "R": [R_th],
            "P": [P_th],
            "F": [F_th],
        }
    )
    # Sanity check
    if FP_th + TP_th != len(detec_conf):
        raise Exception(
            "FP and TP don't add up to the total number of detections"
        )
    elif TP_th + FN_th != len(annot):
        raise Exception(
            "FP and FN don't add up to the total number of annotations"
        )

    if th_idx == 0:
        performance_PRF = tmp_PRF
    else:
        performance_PRF = pd.concat(
            [performance_PRF, tmp_PRF], ignore_index=True
        )

print("s")
# plot PRF curves
fig, ax = plt.subplots(1, 2)
# ax[0].axis('equal')
ax[0].plot(performance_PRF["P"], performance_PRF["R"], "k")
ax[0].set_xlabel("Precision")
ax[0].set_ylabel("Recall")
ax[0].set(xlim=(0, 1), ylim=(0, 1))
ax[0].grid()
ax[0].set_aspect("equal", "box")
fig.tight_layout()

# ax[1].axis('equal')
ax[1].plot(
    performance_PRF["threshold"], performance_PRF["P"], ":k", label="Precision"
)
ax[1].plot(
    performance_PRF["threshold"], performance_PRF["R"], "--k", label="Recall"
)
ax[1].plot(
    performance_PRF["threshold"],
    performance_PRF["F"],
    "k",
    label="$F_" + str(F_beta) + "$-score",
)
ax[1].set_xlabel("Threshold")
ax[1].set_ylabel("Score")
ax[1].set(xlim=[0, 1], ylim=[0, 1])
ax[1].grid()
ax[1].legend()
ax[1].set_aspect("equal", "box")
fig.set_size_inches(8, 4)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "Performance_graph.png"))

# save CSV files
performance_PRF.to_csv(
    os.path.join(out_dir, "Performance_full_dataset.csv"), index=False
)
performance_per_file_count.to_csv(
    os.path.join(out_dir, "Performance_per_file.csv"), index=False
)

# # calculate P, R, F for each threshold
# for th_idx, threshold in enumerate(thresholds):
#     perf_tmp = performance_per_file_count.query(
#         "threshold ==" + str(threshold)
#     )
#     TP = perf_tmp["TP"].sum()
#     FP = perf_tmp["FP"].sum()
#     FN = perf_tmp["FN"].sum()
#     R = TP / (TP + FN)
#     P = TP / (TP + FP)
#     F = (1 + F_beta**2) * ((P * R) / ((F_beta**2 * P) + R))
#     tmp = pd.DataFrame(
#         {
#             "threshold": [threshold],
#             "TP": [TP],
#             "FP": [FP],
#             "FN": [FN],
#             "R": [R],
#             "P": [P],
#             "F": [F],
#         }
#     )
#     # Sanity check
#     if FP+TP != len(detec_conf):
#         raise Exception("FP and TP don't add up to the total number of detections")
#     elif FP[idx]+FN[idx] != len(annot_tmp):
#         raise Exception("FP and FN don't add up to the total number of annotations")

#     if th_idx == 0:
#         performance_PRF = tmp
#     else:
#         performance_PRF = pd.concat([performance_PRF, tmp], ignore_index=True)


print("ss")
# # initialize TP, FP, TP
# TP = np.zeros(
#     (len(annot), len(thresholds))
# )  # True positives (annot x threshold)
# FP = np.zeros(
#     (len(annot), len(thresholds))
# )  # False positives (annot x threshold)
# FN = np.zeros(
#     (len(annot), len(thresholds))
# )  # False negatives (annot x threshold)
# TN = np.zeros(
#     (len(annot), len(thresholds))
# )  # True negatives (annot x threshold)

# # go through each annotation
# for an_idx, an in annot.iterrows():
#     print(an_idx, an["audio_file_name"])

#     # Annotation label, and start/stop stimes
#     an_label = an["label"]
#     an_t1 = an["time_min_offset"]
#     an_t2 = an["time_max_offset"]

#     # if an_label == class_pos:
#     #    print('stop here')

#     # load detection file
#     try:
#         detec = pd.read_csv(
#             os.path.join(
#                 detec_dir,
#                 an["audio_file_name"] + ".wav.chan1.Table.1.selections.txt",
#             ),
#             sep="\t",
#         )
#     except:
#         detec = None  # no detections at all

#     # go through thresholds
#     for th_idx, th in enumerate(thresholds):
#         # print(th)
#         # only keeps detectio above current threshold
#         if detec is not None:
#             detec_th = detec[detec["Confidence"] >= th]
#             if len(detec_th) == 0:
#                 detec_th = None
#         else:
#             detec_th = None

#         # find detections overlapping with annotation
#         is_detec = False
#         if (
#             detec_th is not None
#         ):  # if there are any detections left at this threshold
#             for _, tmp in detec_th.iterrows():
#                 det_t1 = tmp["Begin Time (s)"]
#                 det_t2 = tmp["End Time (s)"]

#                 is_overlap = (
#                     ((det_t1 <= an_t1) & (det_t2 >= an_t2))
#                     | (  # 1- annot inside detec
#                         (det_t1 >= an_t1) & (det_t2 <= an_t2)
#                     )
#                     | (  # 2- detec inside annot
#                         (det_t1 < an_t1) & (det_t2 < an_t2) & (det_t2 > an_t1)
#                     )
#                     | (  # 3- only the end of the detec overlaps with annot
#                         (det_t1 > an_t1) & (det_t1 < an_t2) & (det_t2 > an_t2)
#                     )
#                 )  # 4- only the begining of the detec overlaps with annot

#                 if is_overlap == True:
#                     is_detec = True
#                     break
#         # count TP, FP, FN
#         if (an_label == class_pos) & (is_detec == True):  # TP
#             TP[an_idx, th_idx] = 1
#         if (an_label != class_pos) & (is_detec == False):  # TN
#             TN[an_idx, th_idx] = 1
#         if (an_label == class_pos) & (is_detec == False):  # FN
#             FN[an_idx, th_idx] = 1
#         if (an_label != class_pos) & (is_detec == True):  # FP
#             FP[an_idx, th_idx] = 1

# # sum up
# TP_count = sum(TP)
# TN_count = sum(TN)
# FN_count = sum(FN)
# FP_count = sum(FP)

# # calculate metrics for each trheshold
# P = np.zeros(len(thresholds))
# R = np.zeros(len(thresholds))
# F = np.zeros(len(thresholds))
# for th_idx in range(0, len(thresholds)):
#     P[th_idx] = TP_count[th_idx] / (TP_count[th_idx] + FP_count[th_idx])
#     R[th_idx] = TP_count[th_idx] / (TP_count[th_idx] + FN_count[th_idx])
#     F[th_idx] = (2 * P[th_idx] * R[th_idx]) / (P[th_idx] + R[th_idx])


# # save
# df = pd.DataFrame({"Thresholds": thresholds, "P": P, "R": R, "F": F})
# df.to_csv(os.path.join(detec_dir, "performance.csv"))
# df2 = pd.DataFrame(TP)
# df2 = pd.concat([annot["audio_file_name"], df2], ignore_index=True, axis=1)
# df2.to_csv(os.path.join(detec_dir, "TP.csv"))
# df3 = pd.DataFrame(FP)
# df3 = pd.concat([annot["audio_file_name"], df3], ignore_index=True, axis=1)
# df3.to_csv(os.path.join(detec_dir, "FP.csv"))
# df4 = pd.DataFrame(FN)
# df4 = pd.concat([annot["audio_file_name"], df4], ignore_index=True, axis=1)
# df4.to_csv(os.path.join(detec_dir, "FN.csv"))

# ## Graphs
# plt.plot(P, R, "k")
# plt.xlabel("Precision")
# plt.ylabel("Recall")
# plt.grid()
# plt.xticks(np.arange(0, 1 + 0.02, 0.02))
# plt.yticks(np.arange(0, 1 + 0.02, 0.02))
# plt.ylim([0.8, 1])
# plt.xlim([0.8, 1])
# plt.savefig(os.path.join(detec_dir, "PR_curve.png"))

# plt.figure()
# plt.plot(thresholds, R, ":k", label="Recall")
# plt.plot(thresholds, P, "--k", label="Precision")
# plt.plot(thresholds, F, "k", label="F-score")
# plt.legend()
# plt.xlabel("Threshold")
# plt.ylabel("Performance")
# plt.grid()
# plt.xticks(np.arange(0, 1 + 0.05, 0.05))
# plt.yticks(np.arange(0, 1 + 0.02, 0.02))
# plt.ylim([0.8, 1])
# plt.xlim([thresholds[0], thresholds[-1]])
# plt.savefig(os.path.join(detec_dir, "PRF_curves.png"))
