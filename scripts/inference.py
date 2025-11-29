"""
inference.py - Модуль для инференса обученной модели
"""

import torch
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_model


def get_inference_transform():
    """
    Создание трансформаций для инференса
    """
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def parse_ingredients(ingredients_str):
    """
    Парсинг строки с ингредиентами
    """
    if pd.isna(ingredients_str):
        return []
    return [int(x.replace('ingr_', '')) for x in ingredients_str.split(';') if x]


def encode_ingredients(ingredients_list, num_ingredients):
    """
    Кодирование ингредиентов в multi-hot encoding
    """
    encoding = torch.zeros(num_ingredients)
    for ingr_id in ingredients_list:
        if ingr_id < num_ingredients:
            encoding[ingr_id] = 1.0
    return encoding


def predict_single(model, image_path, ingredients_str, mass, num_ingredients, device='cpu'):
    """
    Предсказание калорийности для одного блюда
    """
    # Загрузка и преобразование изображения
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    transform = get_inference_transform()
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Обработка ингредиентов
    ingredients_list = parse_ingredients(ingredients_str)
    ingredients_tensor = encode_ingredients(ingredients_list, num_ingredients)
    ingredients_tensor = ingredients_tensor.unsqueeze(0).to(device)

    # Масса блюда
    mass_tensor = torch.tensor([mass], dtype=torch.float32).to(device)

    # Предсказание
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor, ingredients_tensor, mass_tensor)

    return prediction.item()


def predict_batch(model, test_loader, device='cpu'):
    """
    Предсказание калорийности для батча данных
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            ingredients = batch['ingredients'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['calories'].to(device)

            predictions = model(images, ingredients, mass)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_predictions), np.array(all_targets)


def analyze_predictions(predictions, targets, top_k=5):
    """
    Анализ предсказаний модели
    """
    errors = np.abs(predictions - targets)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100

    # Индексы наихудших предсказаний
    worst_indices = np.argsort(errors)[-top_k:][::-1]

    results = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'worst_indices': worst_indices,
        'worst_errors': errors[worst_indices],
        'worst_predictions': predictions[worst_indices],
        'worst_targets': targets[worst_indices]
    }

    return results


def visualize_prediction(image_path, real_calories, predicted_calories):
    """
    Визуализация предсказания для одного блюда
    """
    import matplotlib.pyplot as plt

    image = Image.open(image_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(
        f'Реальная калорийность: {real_calories:.2f} kcal\n'
        f'Предсказанная калорийность: {predicted_calories:.2f} kcal\n'
        f'Ошибка: {abs(real_calories - predicted_calories):.2f} kcal',
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('model_best.pth', num_ingredients=1000, device=device)

    image_path = 'data/images/dish_000001/rgb.png'
    ingredients = 'ingr_0000000122;ingr_0000000026;ingr_0000000045;'
    mass = 350.0

    predicted_calories = predict_single(
        model=model,
        image_path=image_path,
        ingredients_str=ingredients,
        mass=mass,
        num_ingredients=1000,
        device=device
    )

    print(f"Предсказанная калорийность: {predicted_calories:.2f} kcal")
