# %% [markdown]
# # 1. Κατεβάζουμε τα αρχεία

# %%


# %%
from gdown import download

download(id='10iQQcGN80wqRMjGeItnklsODP54hHydP', output='model_ferplus.pth', quiet=False)
download(id='1g56Vxvk506MV4mxf3489WI-KBgmKiRLw', output='angry.png', quiet=False)
download(id='1ej3OzvPL_Itck2v3Atln671l-RfrvCss', output='happy.png', quiet=False)
download(id='1-2C4lT5WdAXSleGOG0KHtTeYfyDonhJn', output='neutral.png', quiet=False)

# %% [markdown]
# # 2. Φορτώνουμε το μοντέλο

# %%
import torch
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReshapeAndScale255(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(0)
        return ( x.unsqueeze(1) / 255 ).clamp(0,1)
model = torch.nn.Sequential(
    ReshapeAndScale255(),
    models.shufflenet_v2_x1_0(num_classes = 8)
)
model[1].conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model[1].load_state_dict(torch.load('model_ferplus.pth', map_location=device))
model.eval()

emotion_labels = [
    "neutral",    # 0
    "happy",      # 1
    "surprise",   # 2
    "sad",        # 3
    "angry",      # 4
    "disgust",    # 5
    "fear",       # 6
    "contempt"    # 7
]

# %% [markdown]
# # 3. Φορτώνουμε τη φωτογραφία σαν torch.tensor και τις βοηθητικές συναρτήσεις
# 
# Μια φωτογραφία αντιστοιχεί σε ένα πίνακα με 112 γραμμές και 112 στήλες.
# Κάθε τιμή του πίνακα είναι από 0 εώς 255
# 
# - **loadImage**: Φορτώνει μια εικόνα όταν της δοθεί το path του αρχείου.  
# - **tensorToImage**: Μετατρέπει ένα PyTorch `Tensor` σε κανονική εικόνα
# 

# %%
from PIL import Image
import numpy as np

def loadImage(filename):
    img = Image.open(filename).convert("L").resize( size=(112,112) )
    img_data = np.array(img).astype(np.float32)
    return torch.tensor(img_data)

def tensorToImage(tensor):
    return Image.fromarray(tensor.clamp(0,255).detach().reshape(112,112).numpy().astype(np.uint8))

Xoriginal = loadImage('angry.png')
print(Xoriginal)
tensorToImage(Xoriginal)

# %% [markdown]
# # 4. Αλλάζουμε τη φωτογραφία και βλέπουμε το καινούριο συναίσθημα

# %%
import torch
import torch.nn.functional as F
from time import perf_counter

def sparse_l1_attack(model, image, target_emotion, emotion_labels,
                     max_iters=500,
                     pixels_per_iter=1):
    t1 = perf_counter()

    model.eval()

    target_class = emotion_labels.index(target_emotion)
    adv = image.clone().detach().float()
    original = image.clone().detach().float()

    for iteration in range(max_iters):

        adv.requires_grad_(True)

        output = model(adv.clamp(0,255))
        pred_class = output.argmax(dim=1).item()
        pred_emotion = emotion_labels[pred_class]

        if (iteration % 100 == 0):
          print(f"Iteration {iteration}")
          probabilities = F.softmax(output, dim=1).squeeze()
          print("Model Confidence:")
          for emotion, probability in zip(emotion_labels, probabilities):
              print(f"{probability*100:8.1f}% - {emotion}")
          print(f"Predicted Emotion: {pred_emotion}")

        if pred_class == target_class:
            print(f"Attack succeeded at iteration {iteration}")
            print(f"Dt {perf_counter() - t1}")
            print(f"New emotion: {pred_emotion}")
            break

        loss = F.cross_entropy(output, torch.tensor([target_class]))
        model.zero_grad()
        loss.backward()

        grad = adv.grad.detach()

        # Create a detached copy for modifications to avoid in-place error
        adv_temp = adv.clone().detach()
        adv_temp_flat = adv_temp.view(-1)
        grad_flat = grad.view(-1)

        indices = torch.argsort(grad_flat.abs(), descending=True)

        changed = 0
        with torch.no_grad(): # Ensure these operations don't build a new graph
            for idx in indices:
                if changed >= pixels_per_iter:
                    break

                g = grad_flat[idx]
                if g == 0:
                    continue

                step = -1 if g > 0 else 1
                new_value = adv_temp_flat[idx] + step

                if 0 <= new_value <= 255:
                    adv_temp_flat[idx] = new_value
                    changed += 1

        adv = adv_temp.clone() # Reassign adv with the modified, detached tensor
        adv = adv.detach() # Detach for the next iteration's requires_grad

    distance = (original.round() - adv.round()).abs().sum().item()
    print(f"Final L1 distance = {distance}")
    print(f"Dt {perf_counter() - t1}")

    return adv

# %%
import torch
import torch.nn.functional as F

def generate_pair(model, image, target_emotion, emotion_labels,
                  max_iters=500, pixels_per_iter=1):

    model.eval()

    target_class = emotion_labels.index(target_emotion)

    adv = image.clone().detach().float()
    prev_adv = adv.clone().detach()
    original = image.clone().detach().float()

    for iteration in range(max_iters):
        adv.requires_grad_(True)

        output = model(adv.clamp(0,255))
        pred_class = output.argmax(dim=1).item()

        if (iteration % 100 == 0):
          print(f"Iteration {iteration}")
          probabilities = F.softmax(output, dim=1).squeeze()
          print("Model Confidence:")
          for emotion, probability in zip(emotion_labels, probabilities):
              print(f"{probability*100:8.1f}% - {emotion}")

        if pred_class == target_class:
            print(f"Boundary crossed at iteration {iteration}")
            break

        # Save previous state BEFORE modifying
        prev_adv = adv.clone().detach()

        loss = F.cross_entropy(output, torch.tensor([target_class]))
        model.zero_grad()
        loss.backward()

        grad = adv.grad.detach()

        # Create a detached copy for modifications to avoid in-place error
        adv_temp = adv.clone().detach()
        adv_temp_flat = adv_temp.view(-1)
        grad_flat = grad.view(-1)

        indices = torch.argsort(grad_flat.abs(), descending=True)

        changed = 0
        with torch.no_grad(): # Ensure these operations don't build a new graph
            for idx in indices:
                if changed >= pixels_per_iter:
                    break

                g = grad_flat[idx]
                if g == 0:
                    continue

                step = -1 if g > 0 else 1
                new_value = adv_temp_flat[idx] + step

                if 0 <= new_value <= 255:
                    adv_temp_flat[idx] = new_value
                    changed += 1

        adv = adv_temp.clone() # Reassign adv with the modified, detached tensor
        adv = adv.detach() # Detach for the next iteration's requires_grad

    # Compute exact L1 distance between the two submitted images
    distance = (prev_adv.round() - adv.round()).abs().sum().item()
    print(f"L1 distance between pair = {distance}")

    return prev_adv, adv

# %%
import torch
import torch.nn.functional as F

def untargeted_sparse_attack(model, image, emotion_labels,
                             max_iters=500,
                             pixels_per_iter=1):

    model.eval()

    adv = image.clone().detach().float()
    original = image.clone().detach().float()

    # Get original class (happy)
    with torch.no_grad():
        output = model(adv.clamp(0,255))
        true_class = output.argmax(dim=1)

    print("Original emotion:", emotion_labels[true_class.item()])

    prev_adv = adv.clone().detach()

    for iteration in range(max_iters):
        adv.requires_grad_(True)

        output = model(adv.clamp(0,255))
        pred_class = output.argmax(dim=1)

        if (iteration % 100 == 0):
          print(f"Iteration {iteration}")
          probabilities = F.softmax(output, dim=1).squeeze()
          print("Model Confidence:")
          for emotion, probability in zip(emotion_labels, probabilities):
              print(f"{probability*100:8.1f}% - {emotion}")

        # Stop when class changes
        if pred_class.item() != true_class.item():
            print(f"Class changed at iteration {iteration}")
            print("New emotion:", emotion_labels[pred_class.item()])
            break

        prev_adv = adv.clone().detach()

        loss = F.cross_entropy(output, true_class)
        model.zero_grad()
        loss.backward()

        grad = adv.grad.detach()

        # Create a detached copy for modifications to avoid in-place error
        adv_temp = adv.clone().detach()
        adv_temp_flat = adv_temp.view(-1)
        grad_flat = grad.view(-1)

        indices = torch.argsort(grad_flat.abs(), descending=True)

        changed = 0
        with torch.no_grad(): # Ensure these operations don't build a new graph
            for idx in indices:
                if changed >= pixels_per_iter:
                    break

                g = grad_flat[idx]
                if g == 0:
                    continue

                # IMPORTANT: push AWAY from true class
                step = 1 if g > 0 else -1

                new_value = adv_temp_flat[idx] + step

                if 0 <= new_value <= 255:
                    adv_temp_flat[idx] = new_value
                    changed += 1

        adv = adv_temp.clone() # Reassign adv with the modified, detached tensor
        adv = adv.detach() # Detach for the next iteration's requires_grad

    distance = (original.round() - adv.round()).abs().sum().item()
    print(f"Final L1 distance = {distance}")

    return prev_adv, adv

# %%
#TODO, παράδειγμα παρακάτω
easy_first = loadImage('neutral.png')
easy_second = loadImage('neutral.png')
easy_second[85:87,40:70] = 255

medium_first = loadImage('angry.png')
medium_second = loadImage('angry.png')


hard_first = loadImage('happy.png')
hard_second = loadImage('happy.png')

# %% [markdown]
# Η συνάρτηση `compare_images` θα σας βοηθήσει να βλέπετε γρήγορα τις 2 εικόνες μαζί με τις προβλέψεις του μοντέλου.
# 
# **ΣΗΜΑΝΤΙΚΟ!!**
# 
# Πριν περάσουν στο μοντέλο, οι εικόνες στρογγυλοποιούνται στα κοντινότερα ακέραια pixel values. Πχ, εάν ένα pixel έχει τιμή 225.3, τότε γίνεται 225.

# %%
def compare_images(A, B):
    predictionA = model(A.clamp(0,255).round()).squeeze().softmax(-1)
    predictionB = model(B.clamp(0,255).round()).squeeze().softmax(-1)

    display(tensorToImage(A))
    print("predictions A:")
    for emotion, probability in zip(emotion_labels, predictionA):
        print(f"{probability*100:8.1f}% - {emotion}")

    emotionA = emotion_labels[predictionA.argmax()]
    print(f"emotion A = {emotionA}\n")

    display(tensorToImage(B))
    print("predictions B:")
    for emotion, probability in zip(emotion_labels, predictionB):
        print(f"{probability*100:8.1f}% - {emotion}")

    emotionB = emotion_labels[predictionB.argmax()]
    print(f"emotion B = {emotionB}\n")

    distance = (A.round() - B.round()).abs().sum().int().item()
    print(f"distance = {distance}")


compare_images(easy_first, easy_second)

# %%
target_emotion = "happy"
target_class = emotion_labels.index(target_emotion)

easy_first, easy_second = generate_pair(model, easy_first, target_emotion, emotion_labels, max_iters=5000, pixels_per_iter=2) # emotion_labels

# The following line was adding a white rectangle; remove it to ensure easy_second is solely from the attack.
# easy_second[85:87,40:70] = 255
compare_images(easy_first, easy_second)

# %%
target_emotion = "happy"
target_class = emotion_labels.index(target_emotion)

medium_second = sparse_l1_attack(model, medium_first, target_emotion, emotion_labels, max_iters=10000, pixels_per_iter=5) # emotion_labels

compare_images(medium_first, medium_second)

# %%
hard_first = loadImage('happy.png')
_, hard_first = untargeted_sparse_attack(model, hard_first, emotion_labels, 5000, 2)

# %% [markdown]
# # 5. Αποθήκευση Απαντήσεων για υποβολή στο site

# %%
import json

def toList(tensor):
    return tensor.clamp(0,255).round().int().tolist()

answers = {
    "easy": {
        "first": toList(easy_first),
        "second": toList(easy_second)
    },
    "medium": {
        "second": toList(medium_second),
    },
    "hard": {
        "first": toList(hard_first),
    }
}
with open("answers.json", "w") as f:
    json.dump(answers, f)

# %%
from google.colab import files
files.download('answers.json')


