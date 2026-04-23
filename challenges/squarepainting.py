# %% [markdown]
# # Κατεβάζουμε τα αρχεία

# %%
from gdown import download
download(id='1ecv7SSUCy6v35106oxqf7F4BvZOSdR2L', output='rubiks.jpg')
download(id='1w43BI_6ZngVDD6NHZ4OYE16q0ULTbJal', output='cat.jpg')

# %% [markdown]
# # **Χρωματισμός Τετραγώνου**
# 
#   Θέλουμε να χρωματίσουμε ένα τετράγωνο πλευράς 1. Για να γίνει αυτό, θα χρησιμοποιήσουμε μια συνάρτηση που αντιστοιχίζει σε κάθε σημείο $(x,y)$ με $0 \le x,y \le 1$, τρεις τιμές χρωματικής φωτεινότητας $(r, g, b)$ με $0 \le r, g, b \le 1$ που αντιστοιχούν στην κόκκινη, στην πράσινη και στην μπλέ συνιστώσα.
# 
# <p align="center">
#   <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4xIiBlbmNvZGluZz0idXRmLTgiID8+Cjxzdmcgdmlld0JveD0iOTAgNDAgMzgwIDI3MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxzdHlsZT4KICAgICAgICBsaW5lIHsKICAgICAgICAgICAgc3Ryb2tlOiByZ2IoMTAlLCAxMCUsIDE2JSk7CiAgICAgICAgfQoKICAgICAgICBjaXJjbGUgewogICAgICAgICAgICBmaWxsOiBsaWdodGJsdWU7CiAgICAgICAgICAgIHN0cm9rZTogcmdiKDEwJSwgMTAlLCAxNiUpOwogICAgICAgIH0KCiAgICAgICAgLmdyZWVuIHsKICAgICAgICAgICAgZmlsbDogbWVkaXVtc2VhZ3JlZW47CiAgICAgICAgfQoKICAgICAgICAucmVkIHsKICAgICAgICAgICAgZmlsbDogb3JhbmdlcmVkOwogICAgICAgIH0KCiAgICAgICAgLmJsdWUgewogICAgICAgICAgICBmaWxsOiBkb2RnZXJibHVlOwogICAgICAgIH0KCiAgICAgICAgdGV4dCB7CiAgICAgICAgICAgIGZvbnQtZmFtaWx5OiBBcmlhbCwgSGVsdmV0aWNhLCBzYW5zLXNlcmlmOwogICAgICAgICAgICBmaWxsOiBibGFjazsKICAgICAgICAgICAgZm9udC1zaXplOiAyMHB4OwogICAgICAgICAgICBhbGlnbm1lbnQtYmFzZWxpbmU6IG1pZGRsZTsKICAgICAgICAgICAgdGV4dC1hbmNob3I6IG1pZGRsZTsKICAgICAgICB9CiAgICA8L3N0eWxlPgogICAgPCEtLSBCYWNrZ3JvdW5kIHJlY3RhbmdsZSAtLT4KICAgIDxyZWN0IHg9IjEwMCIgeT0iMTAwIiB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iZGFya2dyYXkiIC8+CgogICAgPCEtLSBBeGVzIC0tPgogICAgPGxpbmUgeDE9IjkwIiB5MT0iMzAwIiB4Mj0iMzUwIiB5Mj0iMzAwIiBzdHJva2U9ImJsYWNrIiAvPgoKICAgIDxsaW5lIHgxPSIzNDAiIHkxPSIyOTUiIHgyPSIzNTAiIHkyPSIzMDAiIHN0cm9rZT0iYmxhY2siIC8+CiAgICA8bGluZSB4MT0iMzQwIiB5MT0iMzA1IiB4Mj0iMzUwIiB5Mj0iMzAwIiBzdHJva2U9ImJsYWNrIiAvPgoKICAgIDxsaW5lIHgxPSIxMDAiIHkxPSIzMTAiIHgyPSIxMDAiIHkyPSI1MCIgc3Ryb2tlPSJibGFjayIgLz4KICAgIDxsaW5lIHgxPSIgOTUiIHkxPSI2MCIgeDI9IjEwMCIgeTI9IjUwIiBzdHJva2U9ImJsYWNrIiAvPgogICAgPGxpbmUgeDE9IjEwNSIgeTE9IjYwIiB4Mj0iMTAwIiB5Mj0iNTAiIHN0cm9rZT0iYmxhY2siIC8+CgogICAgPCEtLSBDb29yZGluYXRlIGxhYmVsIC0tPgogICAgPHRleHQgeD0iMjAwIiB5PSIyMDAiIGZvbnQtc2l6ZT0iMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9ImJsYWNrIj4oeCwgeSk8L3RleHQ+CgogICAgPCEtLSBSZWQgZG90IC0tPgogICAgPGNpcmNsZSBjeD0iMjQwIiBjeT0iMjAwIiByPSI1IiBmaWxsPSJyZWQiIC8+CgogICAgPCEtLSBBcnJvdyBmcm9tIGRvdCB0byBSR0IgLS0+CiAgICA8bGluZSB4MT0iMjQ1IiB5MT0iMjAwIiB4Mj0iNDAwIiB5Mj0iMjAwIiBzdHJva2U9ImJsYWNrIiAvPgogICAgPCEtLSBBcnJvd2hlYWQgLS0+CiAgICA8bGluZSB4MT0iMzkwIiB5MT0iMTk1IiB4Mj0iNDAwIiB5Mj0iMjAwIiBzdHJva2U9ImJsYWNrIiAvPgogICAgPGxpbmUgeDE9IjM5MCIgeTE9IjIwNSIgeDI9IjQwMCIgeTI9IjIwMCIgc3Ryb2tlPSJibGFjayIgLz4KCiAgICA8IS0tIFJHQiBvdXRwdXQgdGV4dCAtLT4KICAgIDx0ZXh0IHg9IjQxNyIgeT0iMjAwIiBmb250LXNpemU9IjIwIiBjbGFzcz0iYmxhY2siPig8L3RleHQ+CiAgICA8dGV4dCB4PSI0MjQiIHk9IjIwMCIgZm9udC1zaXplPSIyMCIgY2xhc3M9InJlZCI+cjwvdGV4dD4KICAgIDx0ZXh0IHg9IjQyOSIgeT0iMjAwIiBmb250LXNpemU9IjIwIiBmaWxsPSJibGFjayI+LDwvdGV4dD4KICAgIDx0ZXh0IHg9IjQzNyIgeT0iMjAwIiBmb250LXNpemU9IjIwIiBjbGFzcz0iZ3JlZW4iPmc8L3RleHQ+CiAgICA8dGV4dCB4PSI0NDUiIHk9IjIwMCIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siPiw8L3RleHQ+CiAgICA8dGV4dCB4PSI0NTQiIHk9IjIwMCIgZm9udC1zaXplPSIyMCIgY2xhc3M9ImJsdWUiPmI8L3RleHQ+CiAgICA8dGV4dCB4PSI0NjMiIHk9IjIwMCIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siPik8L3RleHQ+Cjwvc3ZnPg==" alt="Εικόνα" width="550" height="300">
# </p>
# 
# 
#   Για παράδειγμα, η συνάρτηση $f(x,y) = (x,y,0)$ βάφει:
# - το σημείο $(x,y) = (0,0)$ μαύρο, δλδ $(r,g,b) = (0,0,0)$,
# - το σημείο $(x,y) = (1,0)$ κόκκινο, δλδ $(r,g,b) = (1,0,0)$,
# - το σημείο $(x,y) = (0,1)$ πράσινο, δλδ $(r,g,b) = (0,1,0)$,
# - το σημείο $(x,y) = (1,1)$ κίτρινο, δλδ $(r,g,b) = (1,1,0)$ συνδυάζοντας την κόκκινη και πράσινη συνιστώσα
# - κάθε άλλο σημείο ένα μείγμα αυτών των χρωμάτων.
# 
# 
# 
# Η μέθοδος `function2image` μετατρέπει μια δοσμένη συνάρτηση σε εικόνα.
# 

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import types
import math

def function2image(f, grid_size=500):
    x_vals = torch.linspace(0, 1, grid_size)
    y_vals = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    if isinstance(f, types.FunctionType):
        output = torch.tensor([f(x.item(), y.item()) for x, y in zip(X_flat, Y_flat)])
    elif isinstance(f, torch.nn.Module):
        inp = torch.column_stack( (X_flat, Y_flat) )
        output = f( inp ).detach()
    else:
        print("Unsupported function")
        return

    output_image = output.clamp(0, 1).view(grid_size, grid_size, 3)

    plt.imshow(output_image.numpy(), origin="lower")
    plt.axis('off')
    plt.show()

def basic_f(x, y):
    return (x, y, 0)

def easy_f(x, y):
    return (0, 0, abs(x+y - 1))

def medium_f(x, y):
    return ((math.sin(x*15)) * 0.5 +0.5,(math.cos(y*15)) * 0.5+0.5,0)

function2image(basic_f)

# %% [markdown]
# ## Πιο δύσκολες συναρτήσεις με βάση φωτογραφίες

# %%
from PIL import Image
from torchvision import transforms as T

advanced_img = T.ToTensor()(Image.open("rubiks.jpg").convert("RGB"))
def advanced_f(x, y):
    h, w = advanced_img.shape[1], advanced_img.shape[2]
    xi = min(math.floor(x * w), w - 1)
    yi = min(math.floor((1-y) * h), h - 1)
    return advanced_img[:, yi, xi].tolist()

hard_img = T.ToTensor()(Image.open("cat.jpg").convert("RGB"))
def hard_f(x, y):
    h, w = hard_img.shape[1], hard_img.shape[2]
    xi = min(math.floor(x * w), w - 1)
    yi = min(math.floor((1-y) * h), h - 1)
    return hard_img[:, yi, xi].tolist()


function2image(hard_f)

# %% [markdown]
# ## Η άσκηση
# 
# Για την άσκηση θα χρησιμοποιήσουμε μια δοσμένη αρχιτεκτονική νεωρωνικού δικτύου ως συνάρτηση για να χρωματίσουμε το τετράγωνο. Το νευρωνικό δέχεται 2 εισόδους τις συντεταγμένες $(x,y)$ και βγάζει ως έξοδο 3 τιμές που αντιστοιχούν στις φωτεινότητες rgb. Στόχος σας είναι να εκπαιδεύσετε μια τέτοια αρχιτεκτονική για κάθε εικόνα έτσι ώστε να οδηγήσει σε όσο πιο πιστή αναπαράσταση της εικόνας μπορείτε.
# 
# 
# <p align="center">
#   <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4xIiBlbmNvZGluZz0idXRmLTgiID8+CjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczpldj0iaHR0cDovL3d3dy53My5vcmcvMjAwMS94bWwtZXZlbnRzIgogICAgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIGJhc2VQcm9maWxlPSJ0aW55IiB2ZXJzaW9uPSIxLjIiIHZpZXdCb3g9IjEwMCA1MCA1NTAgMzAwIj4KICAgIDxzdHlsZT4KICAgICAgICBsaW5lIHsKICAgICAgICAgICAgc3Ryb2tlOiByZ2IoMTAlLDEwJSwxNiUpOwogICAgICAgIH0KICAgICAgICBjaXJjbGUgewogICAgICAgICAgICBmaWxsOiBsaWdodGJsdWU7CiAgICAgICAgICAgIHN0cm9rZTogcmdiKDEwJSwxMCUsMTYlKTsKICAgICAgICB9CiAgICAgICAgLmdyZWVuIHtmaWxsOiBtZWRpdW1zZWFncmVlbjt9CiAgICAgICAgLnJlZCB7ZmlsbDpvcmFuZ2VyZWQ7fQogICAgICAgIC5ibHVle2ZpbGw6ZG9kZ2VyYmx1ZTt9CiAgICAgICAgdGV4dCB7CiAgICAgICAgICAgIGZvbnQtZmFtaWx5OiBBcmlhbCwgSGVsdmV0aWNhLCBzYW5zLXNlcmlmOwogICAgICAgICAgICBmaWxsOmJsYWNrOwogICAgICAgICAgICBmb250LXNpemU6IDIwcHg7CiAgICAgICAgICAgIGFsaWdubWVudC1iYXNlbGluZTogbWlkZGxlOwogICAgICAgICAgICB0ZXh0LWFuY2hvcjogbWlkZGxlOwogICAgICAgIH0KICAgIDwvc3R5bGU+CiAgICA8ZGVmcyAvPgogICAgPGxpbmUgeDE9IjE1MCIgeDI9IjMwMCIgeTE9IjE3MCIgeTI9IjExMCIgLz4KICAgIDxsaW5lIHgxPSIxNTAiIHgyPSIzMDAiIHkxPSIxNzAiIHkyPSIxNzAiIC8+CiAgICA8bGluZSB4MT0iMTUwIiB4Mj0iMzAwIiB5MT0iMTcwIiB5Mj0iMjMwIiAvPgogICAgPGxpbmUgeDE9IjE1MCIgeDI9IjMwMCIgeTE9IjE3MCIgeTI9IjI5MCIgLz4KICAgIDxsaW5lIHgxPSIxNTAiIHgyPSIzMDAiIHkxPSIyMzAiIHkyPSIxMTAiIC8+CiAgICA8bGluZSB4MT0iMTUwIiB4Mj0iMzAwIiB5MT0iMjMwIiB5Mj0iMTcwIiAvPgogICAgPGxpbmUgeDE9IjE1MCIgeDI9IjMwMCIgeTE9IjIzMCIgeTI9IjIzMCIgLz4KICAgIDxsaW5lIHgxPSIxNTAiIHgyPSIzMDAiIHkxPSIyMzAiIHkyPSIyOTAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMTEwIiB5Mj0iODAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMTEwIiB5Mj0iMTQwIiAvPgogICAgPGxpbmUgeDE9IjMwMCIgeDI9IjQ1MCIgeTE9IjExMCIgeTI9IjIwMCIgLz4KICAgIDxsaW5lIHgxPSIzMDAiIHgyPSI0NTAiIHkxPSIxMTAiIHkyPSIyNjAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMTEwIiB5Mj0iMzIwIiAvPgogICAgPGxpbmUgeDE9IjMwMCIgeDI9IjQ1MCIgeTE9IjE3MCIgeTI9IjgwIiAvPgogICAgPGxpbmUgeDE9IjMwMCIgeDI9IjQ1MCIgeTE9IjE3MCIgeTI9IjE0MCIgLz4KICAgIDxsaW5lIHgxPSIzMDAiIHgyPSI0NTAiIHkxPSIxNzAiIHkyPSIyMDAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMTcwIiB5Mj0iMjYwIiAvPgogICAgPGxpbmUgeDE9IjMwMCIgeDI9IjQ1MCIgeTE9IjE3MCIgeTI9IjMyMCIgLz4KICAgIDxsaW5lIHgxPSIzMDAiIHgyPSI0NTAiIHkxPSIyMzAiIHkyPSI4MCIgLz4KICAgIDxsaW5lIHgxPSIzMDAiIHgyPSI0NTAiIHkxPSIyMzAiIHkyPSIxNDAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMjMwIiB5Mj0iMjAwIiAvPgogICAgPGxpbmUgeDE9IjMwMCIgeDI9IjQ1MCIgeTE9IjIzMCIgeTI9IjI2MCIgLz4KICAgIDxsaW5lIHgxPSIzMDAiIHgyPSI0NTAiIHkxPSIyMzAiIHkyPSIzMjAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMjkwIiB5Mj0iODAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMjkwIiB5Mj0iMTQwIiAvPgogICAgPGxpbmUgeDE9IjMwMCIgeDI9IjQ1MCIgeTE9IjI5MCIgeTI9IjIwMCIgLz4KICAgIDxsaW5lIHgxPSIzMDAiIHgyPSI0NTAiIHkxPSIyOTAiIHkyPSIyNjAiIC8+CiAgICA8bGluZSB4MT0iMzAwIiB4Mj0iNDUwIiB5MT0iMjkwIiB5Mj0iMzIwIiAvPgogICAgPGxpbmUgeDE9IjQ1MCIgeDI9IjYwMCIgeTE9IjgwIiB5Mj0iMTQwIiAvPgogICAgPGxpbmUgeDE9IjQ1MCIgeDI9IjYwMCIgeTE9IjgwIiB5Mj0iMjAwIiAvPgogICAgPGxpbmUgeDE9IjQ1MCIgeDI9IjYwMCIgeTE9IjgwIiB5Mj0iMjYwIiAvPgogICAgPGxpbmUgeDE9IjQ1MCIgeDI9IjYwMCIgeTE9IjE0MCIgeTI9IjE0MCIgLz4KICAgIDxsaW5lIHgxPSI0NTAiIHgyPSI2MDAiIHkxPSIxNDAiIHkyPSIyMDAiIC8+CiAgICA8bGluZSB4MT0iNDUwIiB4Mj0iNjAwIiB5MT0iMTQwIiB5Mj0iMjYwIiAvPgogICAgPGxpbmUgeDE9IjQ1MCIgeDI9IjYwMCIgeTE9IjIwMCIgeTI9IjE0MCIgLz4KICAgIDxsaW5lIHgxPSI0NTAiIHgyPSI2MDAiIHkxPSIyMDAiIHkyPSIyMDAiIC8+CiAgICA8bGluZSB4MT0iNDUwIiB4Mj0iNjAwIiB5MT0iMjAwIiB5Mj0iMjYwIiAvPgogICAgPGxpbmUgeDE9IjQ1MCIgeDI9IjYwMCIgeTE9IjI2MCIgeTI9IjE0MCIgLz4KICAgIDxsaW5lIHgxPSI0NTAiIHgyPSI2MDAiIHkxPSIyNjAiIHkyPSIyMDAiIC8+CiAgICA8bGluZSB4MT0iNDUwIiB4Mj0iNjAwIiB5MT0iMjYwIiB5Mj0iMjYwIiAvPgogICAgPGxpbmUgeDE9IjQ1MCIgeDI9IjYwMCIgeTE9IjMyMCIgeTI9IjE0MCIgLz4KICAgIDxsaW5lIHgxPSI0NTAiIHgyPSI2MDAiIHkxPSIzMjAiIHkyPSIyMDAiIC8+CiAgICA8bGluZSB4MT0iNDUwIiB4Mj0iNjAwIiB5MT0iMzIwIiB5Mj0iMjYwIiAvPgogICAgPGNpcmNsZSBjeD0iMTUwIiBjeT0iMTcwIiByPSIxNSIgLz48dGV4dCAgeD0iMTUwIiB5PSIxNzAiPng8L3RleHQ+CiAgICA8Y2lyY2xlIGN4PSIxNTAiIGN5PSIyMzAiIHI9IjE1IiAvPjx0ZXh0ICB4PSIxNTAiIHk9IjIzMCI+eTwvdGV4dD4KICAgIDxjaXJjbGUgY3g9IjMwMCIgY3k9IjExMCIgcj0iMTUiIC8+CiAgICA8Y2lyY2xlIGN4PSIzMDAiIGN5PSIxNzAiIHI9IjE1IiAvPgogICAgPGNpcmNsZSBjeD0iMzAwIiBjeT0iMjMwIiByPSIxNSIgLz4KICAgIDxjaXJjbGUgY3g9IjMwMCIgY3k9IjI5MCIgcj0iMTUiIC8+CiAgICA8Y2lyY2xlIGN4PSI0NTAiIGN5PSI4MCIgcj0iMTUiIC8+CiAgICA8Y2lyY2xlIGN4PSI0NTAiIGN5PSIxNDAiIHI9IjE1IiAvPgogICAgPGNpcmNsZSBjeD0iNDUwIiBjeT0iMjAwIiByPSIxNSIgLz4KICAgIDxjaXJjbGUgY3g9IjQ1MCIgY3k9IjI2MCIgcj0iMTUiIC8+CiAgICA8Y2lyY2xlIGN4PSI0NTAiIGN5PSIzMjAiIHI9IjE1IiAvPgogICAgPGNpcmNsZSBjeD0iNjAwIiBjeT0iMTQwIiBjbGFzcz0icmVkIiByPSIxNSIgLz48dGV4dCAgeD0iNjAwIiB5PSIxNDAiPnI8L3RleHQ+CiAgICA8Y2lyY2xlIGN4PSI2MDAiIGN5PSIyMDAiIGNsYXNzPSJncmVlbiIgcj0iMTUiIC8+PHRleHQgIHg9IjYwMCIgeT0iMjAwIj5nPC90ZXh0PgogICAgPGNpcmNsZSBjeD0iNjAwIiBjeT0iMjYwIiBjbGFzcz0iYmx1ZSIgcj0iMTUiIC8+PHRleHQgIHg9IjYwMCIgeT0iMjYwIj5iPC90ZXh0Pgo8L3N2Zz4=" alt="Εικόνα" width="550" height="500">
# </p>
# 
# 
#  Παρακάτω δίνεται η αρχιτεκτονική και η οπτικοποίηση του νευρωνικού με τυχαία αρχικοποιημένα βάρη.

# %%
# Μην πειράξετε την αρχιτεκτονική
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def Net():
    return nn.Sequential(
        nn.Linear(2, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )

net = Net()
function2image(net)

# %%
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

def train_model(model, train_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    total_loss = 0
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch%10==0:
            print(f"epoch {epoch}\loss: {total_loss/len(train_loader)}")

# %%
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def generate_ds(f, samples=50):
    steps = np.linspace(0, 1, samples)
    inputs = []
    targets = []
    for i in steps:
        for j in steps:
            inputs.append([i, j])
            targets.append(f(i, j))

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Keep inputs and RGB targets separate.
    # The notebook bug came from trying to stack tensors of shape (N, 2) and (N, 3).
    return TensorDataset(inputs, targets)

# %%
ds = generate_ds(basic_f, 100)


# %%
# TODO: Υλοποιήστε τον κώδικα εκπαίδευσης του νευρωνικού για κάθε μία από τις συναρτήσεις στόχους

tasks = {
    "basic":    basic_f,
    "easy":     easy_f,
    "medium":   medium_f,
    "advanced": advanced_f,
    "hard":     hard_f
}
epochs = {
    "basic":    100,
    "easy":     100,
    "medium":   200,
    "advanced": 200,
    "hard":     200
}
ds_sizes = {
    "basic":    100,
    "easy":     100,
    "medium":   200,
    "advanced": 200,
    "hard":     200
}
nets = {}
for task in tasks:
    print(f"Task: {task}")
    net = Net()
    ds = generate_ds(tasks[task], ds_sizes[task])
    dataloader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_model(net, dataloader, epochs[task])
    nets[task] = net
    function2image(net.to("cpu"))

    # TODO: Εκπαιδεύστε το νευρωνικό για κάθε task

# %% [markdown]
# ## Υποβολή
# 
# Παίρνουμε τα βάρη του νευρωνικού για υποβολή στο site.

# %%
import json
def export_weights(net):
    layers = []
    i = 0
    while 2 * i < len(net):
        layer = net[2 * i]
        if isinstance(layer, torch.nn.Linear):
            weight_data = layer.weight.data.cpu().numpy()  # Shape: (num_outputs, num_inputs)
            bias_data = layer.bias.data.cpu().numpy()  # Shape: (num_outputs,)

            weights = weight_data.tolist()
            bias = bias_data.tolist()

            layers.append({
                "weights": weights,
                "bias": bias
            })
        i += 1

    return {"layers": layers}

def export_all_levels(nets, filename="answer.json"):
    result = {}

    for level in ["basic", "easy", "medium", "advanced", "hard"]:
        if level in nets:
            result[level] = export_weights(nets[level])

    with open(filename, 'w') as f:
        json.dump(result, f)

export_all_levels(nets)

# %%



