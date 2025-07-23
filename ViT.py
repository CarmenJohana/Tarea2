import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms

class ViT(nn.Module):
    def __init__(self, img_width, img_channels, patch_size, d_model, num_heads, num_layers, num_classes, ff_dim):
        super().__init__()

        self.patch_size = patch_size

        # Capa para proyectar cada patch plano (de tamaño patch_size x patch_size x canales) a un espacio de dimensión d_model
        self.patch_embedding = nn.Linear(img_channels * patch_size * patch_size, d_model)

        # Token de clasificación (CLS) aprendible, que se concatena a la secuencia de embeddings de parches
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Embedding de posición aprendido (para incluir información posicional en la secuencia de patches + cls token)
        self.position_embedding = nn.Parameter(
            torch.rand(1, (img_width // patch_size) * (img_width // patch_size) + 1, d_model)
        )
        # Definición de una capa de encoder de Transformer (usa batch_first=True para que el batch esté en la primera dimensión)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        # Empaquetamos múltiples capas del encoder para crear el bloque completo de Transformer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Capa final de clasificación: mapea el embedding del CLS token a la cantidad de clases
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        N, C, H, W = x.shape # Batch size, canales, alto y ancho

        # Dividir la imagen en patches (usando unfold para extraer sub-bloques sin superposición)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(N, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, C * self.patch_size * self.patch_size)
        # Resultado: (N, num_patches, tamaño_patch_aplanado)

        # Proyectar cada patch a un vector de dimensión d_model
        x = self.patch_embedding(x)

        # Repetir el token CLS para cada imagen del batch y concatenarlo al inicio de la secuencia
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Sumar embeddings de posición a cada token de la secuencia
        x = x + self.position_embedding

        # Pasar la secuencia completa por el encoder Transformer
        x = self.transformer_encoder(x)

        # Extraer la salida del token CLS (posición 0), que contiene la representación global de la imagen
        x = x[:, 0]

        # Pasar el embedding del CLS token por la capa final para obtener las predicciones
        x = self.fc(x)

        return x
    

batch_size = 128
lr = 3e-4
num_epochs = 15

img_width = 28
img_channels = 1
num_classes = 100
patch_size = 7
embedding_dim = 64
ff_dim = 2048
num_heads = 8
num_layers = 3
weight_decay = 1e-4

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)

saved_outputs = []  # lista para guardar tuplas (epoch, imágenes originales, reconstrucciones o predicciones)

# Selección automática de dispositivo (GPU si está disponible, si no CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"{device=}")
# Inicialización del modelo ViT con los parámetros definidos
model = ViT(
    img_width=img_width,
    img_channels=img_channels,
    patch_size=patch_size,
    d_model=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    ff_dim=ff_dim,
).to(device)

# Optimizador Adam con tasa de aprendizaje y regularización L2
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Bucle de entrenamiento por época
for epoch in range(num_epochs):
    losses = []
    total_train = 0
    correct_train = 0

    model.train()# Modo entrenamiento
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)

        pred = model(img)
        loss = F.cross_entropy(pred, label)

        pred_class = torch.argmax(pred, dim=1)
        correct_train += (pred_class == label).sum().item()
        total_train += pred.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Métricas de entrenamiento
    print(f"[{epoch=}] Train Loss: {sum(losses):.4f} | Train Acc: {correct_train / total_train:.4f}")

    # Evaluación en el conjunto de prueba
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            pred = torch.argmax(model(img), dim=1).cpu()

            correct += (pred == label).sum().item()
            total += pred.shape[0]
        # Guardar predicciones cada 4 épocas
        if epoch % 4 == 0:
            for img, label in test_loader:
                img = img.to(device)
                pred = model(img)
                pred_class = torch.argmax(pred, dim=1).cpu()
                saved_outputs.append((epoch, img.cpu(), pred_class))
                break  # solo guardamos un batch por época
    # Precisión en el conjunto de prueba
    print(f"          Test Acc: {correct / total:.4f}")

for epoch, imgs, preds in saved_outputs:
    plt.figure(figsize=(9, 2.5))
    plt.suptitle(f"Epoch {epoch}", fontsize=14)

    for i in range(min(9, len(imgs))):
        plt.subplot(1, 9, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.title(f"Clase: {preds[i].item()}", fontsize=8) # Predicción
        plt.axis("off")



