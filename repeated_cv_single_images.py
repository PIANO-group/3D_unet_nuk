import os
import statistics
import tkinter
from tkinter import filedialog

from config import sessions_path


def fun_file_selector(request_string: str, selected_files_list: list, search_directory):
    filepath_string = filedialog.askopenfilename(initialdir=search_directory, title=request_string)

    if len(filepath_string) > 0:
        selected_files_list.append(filepath_string)
        fun_file_selector('Select the next file or Cancel to end',
                               selected_files_list,
                               os.path.dirname(filepath_string))

    return selected_files_list


os.chdir(sessions_path)

root = tkinter.Tk()
root.withdraw()


cv_summary_filepaths = fun_file_selector("Choose cv session", [], sessions_path)

print(cv_summary_filepaths)

scores = []

for filepath in cv_summary_filepaths[:]:
    with open(filepath) as file:
        content = file.readlines()
        for line_nr, line in enumerate(content):
            if "Over all images of every test set:" in line:
                line_begin = line_nr
    score_lines = content[line_begin+4:]
    scores.extend([score_line.split("\t\t")[1] for score_line in score_lines])

print("Nr of scores:", len(scores))
scores = [float(score.split("\n")[0]) for score in scores]

print(scores)

fold_cv = os.path.basename(cv_summary_filepaths[0]).split("_")[1]
output_filename = str(len(cv_summary_filepaths)) + "_repeated_" + str(fold_cv) + "_fold.txt"
j = 1
while (os.path.isfile(output_filename)):
    if j == 1:
        output_filename = output_filename.split(".")[0] + "_" + str(j) + ".txt"
    else:
        output_filename = output_filename.rsplit("_", 1)[0] + "_" + str(j) + ".txt"
    j += 1

with open(output_filename, "w") as file:
    content_out = ["Nr of scores: " + str(len(scores)) + "\n\n"]

    content_out.append("cv's used:\n")
    for filepath in cv_summary_filepaths:
        content_out.append(os.path.basename(filepath) + "\n")

    content_out.append("\nover all images:\n")
    content_out.append("dice: \t" + str(statistics.mean(scores)) + "\n")
    content_out.append("std: \t" + str(statistics.pstdev(scores)))

    file.writelines(content_out)
