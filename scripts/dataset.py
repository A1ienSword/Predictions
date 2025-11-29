"""
dataset.py - Модуль для работы с данными
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FoodCalorieDataset(Dataset):
    """
    Кастомный Dataset для работы с датасетом калорийности блюд.
    """

    def __init__(self, dish_df, ingredients_df, img_dir, transform=None, mode='train'):
        self.data = dish_df[dish_df['split'] == mode].reset_index(drop=True)
        self.ingredients_df = ingredients_df
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        self.ingr_dict = dict(zip(ingredients_df['id'], ingredients_df['ingr']))

    def __len__(self):
        return len(self.data)

    def parse_ingredients(self, ingredients_str):
        """
        Парсинг строки с ингредиентами в список ID
        """
        if pd.isna(ingredients_str):
            return []

        # Убираем префикс "ingr_" и оставляем только числа
        ingr_list = [int(x.replace('ingr_', '')) for x in ingredients_str.split(';') if x]
        return ingr_list

    def encode_ingredients(self, ingredients_list, max_ingredients=50):
        """
        Кодирование списка ингредиентов в multi-hot encoding
        """
        num_ingredients = len(self.ingredients_df)
        encoding = torch.zeros(num_ingredients)

        for ingr_id in ingredients_list:
            if ingr_id < num_ingredients:
                encoding[ingr_id] = 1.0

        return encoding

    def __getitem__(self, idx):
        """
        Получение одного примера из датасета
        """
        row = self.data.iloc[idx]
        dish_id = row['dish_id']
        img_path = os.path.join(self.img_dir, dish_id, 'rgb.png')

        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Создаем черное изображение как fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Применяем трансформации (аугментации)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Если трансформации нет, преобразуем в тензор вручную
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Парсинг ингредиентов
        ingredients_list = self.parse_ingredients(row['ingredients'])
        ingredients_encoding = self.encode_ingredients(ingredients_list)

        # Получаем табличные данные
        total_mass = torch.tensor(row['total_mass'], dtype=torch.float32)
        total_calories = torch.tensor(row['total_calories'], dtype=torch.float32)

        return {
            'image': image,
            'ingredients': ingredients_encoding,
            'mass': total_mass,
            'calories': total_calories
        }


def get_transforms(mode='train'):
    """
    Создание трансформаций для аугментации данных
    """
    if mode == 'train':
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def create_dataloaders(dish_csv_path, ingredients_csv_path, img_dir, batch_size=32, num_workers=4):
    """
    Создание DataLoader'ов для обучения и тестирования
    """
    # Загрузка CSV файлов
    dish_df = pd.read_csv(dish_csv_path)
    ingredients_df = pd.read_csv(ingredients_csv_path)

    # Создание трансформаций
    train_transform = get_transforms(mode='train')
    test_transform = get_transforms(mode='test')

    # Создание датасетов
    train_dataset = FoodCalorieDataset(
        dish_df=dish_df,
        ingredients_df=ingredients_df,
        img_dir=img_dir,
        transform=train_transform,
        mode='train'
    )

    test_dataset = FoodCalorieDataset(
        dish_df=dish_df,
        ingredients_df=ingredients_df,
        img_dir=img_dir,
        transform=test_transform,
        mode='test'
    )

    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader, ingredients_df
