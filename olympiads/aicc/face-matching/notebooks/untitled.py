"""model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",backend="torchvision")

imgs = [f[:3] for f in os.listdir(IMAGE_PATH) if f.endswith(".jpg")]

for idx in tqdm(imgs):
    img = the_correct_img_from_idx
    procc = processor(img, return_tensors="pt")
    print(procc)
    emb = model(**procc)
"""