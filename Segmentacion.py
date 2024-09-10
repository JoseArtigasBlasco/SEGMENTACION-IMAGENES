import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Cargar el modelo preentrenado DeepLabV3 con ResNet-101
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Definir las transformaciones necesarias para la imagen de entrada
preprocess = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar la imagen
input_image = Image.open("studying-5831644_640.jpg")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Crear un batch con una única imagen

# Verificar si hay una GPU disponible y mover el modelo y el batch a la GPU si es posible
if torch.cuda.is_available():
    model = model.to('cuda')
    input_batch = input_batch.to('cuda')

# Evaluar el modelo sin calcular los gradientes
with torch.no_grad():
    output = model(input_batch)['out'][0]

# Procesar la salida
output_predictions = output.argmax(0)

# Visualizar la imagen original y la segmentada
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# Imagen original
ax[0].imshow(input_image)
ax[0].set_title('Imagen Original')
ax[0].axis('off')

# Imagen segmentada
ax[1].imshow(output_predictions.cpu().numpy())
ax[1].set_title('Imagen Segmentada')
ax[1].axis('off')

plt.show()