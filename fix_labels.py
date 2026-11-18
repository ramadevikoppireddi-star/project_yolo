import os

folder = "dataset/train/labels"

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            parts[0] = "0"
            new_lines.append(" ".join(parts))

    with open(path, "w") as f:
        f.write("\n".join(new_lines))

print("Labels fixed!")