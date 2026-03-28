from datasets import load_dataset

DATASET_NAME = "Mitsua/art-museums-pd-440k"
ds = load_dataset(DATASET_NAME, split="train", streaming=True)

print("Inspecting first item keys:")
for item in ds:
    print(item.keys())
    print(item)
    break
