"""
utils.py - Утилиты для обучения модели
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import random
import yaml
from tqdm import tqdm
import os
from scripts.dataset import create_dataloaders

def set_seed(seed=42):
    """
    Установка seed для воспроизводимости результатов
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalFoodCalorieModel(nn.Module):
    """
    Мультимодальная модель для предсказания калорийности блюд.
    """

    def __init__(self, num_ingredients, pretrained=True):
        super(MultimodalFoodCalorieModel, self).__init__()
        # Загружаем предобученную модель
        efficientnet = models.efficientnet_b0(pretrained=pretrained)

        # Берем все слои кроме последнего (classifier)
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])

        # Размерность выхода EfficientNet-B0 после GlobalAvgPool: 1280
        self.image_feature_dim = 1280

        # Входные данные: multi-hot encoding ингредиентов + масса блюда
        tabular_input_dim = num_ingredients + 1  # +1 для массы

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.tabular_feature_dim = 128

        # Объединяем признаки из изображений и табличных данных
        fusion_input_dim = self.image_feature_dim + self.tabular_feature_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, image, ingredients, mass):
        """
        Forward pass модели
        """
        # 1. Извлечение признаков из изображения
        image_features = self.image_encoder(image)
        # EfficientNet уже применяет GlobalAvgPool, получаем [batch, 1280, 1, 1]
        image_features = image_features.flatten(1)  # [batch, 1280]

        # 2. Обработка табличных данных
        mass = mass.unsqueeze(1) if mass.dim() == 1 else mass  # [batch, 1]
        tabular_input = torch.cat([ingredients, mass], dim=1)  # [batch, num_ingr + 1]
        tabular_features = self.tabular_encoder(tabular_input)  # [batch, 128]

        # 3. Объединение признаков
        combined = torch.cat([image_features, tabular_features], dim=1)
        fused_features = self.fusion(combined)

        # 4. Предсказание калорийности
        predictions = self.regressor(fused_features)

        return predictions.squeeze(1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Обучение модели на одной эпохе
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')

    for batch in pbar:
        images = batch['image'].to(device)
        ingredients = batch['ingredients'].to(device)
        mass = batch['mass'].to(device)
        targets = batch['calories'].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images, ingredients, mass)

        # Вычисление loss
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Статистика
        total_loss += loss.item()
        num_batches += 1

        # Обновление progress bar
        pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Валидация модели
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')

        for batch in pbar:
            images = batch['image'].to(device)
            ingredients = batch['ingredients'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['calories'].to(device)

            # Forward pass
            predictions = model(images, ingredients, mass)

            # Вычисление метрик
            loss = criterion(predictions, targets)
            mae = torch.abs(predictions - targets).mean()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            num_samples += batch_size

            pbar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'mae': f'{mae.item():.2f}'
            })

    avg_loss = total_loss / num_samples
    avg_mae = total_mae / num_samples

    return avg_loss, avg_mae


def train(config_path):
    """
    Основная функция обучения модели
    """
    # Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("НАЧАЛО ОБУЧЕНИЯ")
    print(f"Конфигурация: {config_path}")

    # Установка seed для воспроизводимости
    set_seed(config['seed'])

    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")    

    # Создание DataLoader'ов
    print("Загрузка данных...")
    train_loader, test_loader, ingredients_df = create_dataloaders(
        dish_csv_path=config['data']['dish_csv'],
        ingredients_csv_path=config['data']['ingredients_csv'],
        img_dir=config['data']['img_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    num_ingredients = len(ingredients_df)
    print(f"Количество ингредиентов: {num_ingredients}")

    # Создание модели
    print("Создание модели...")
    model = MultimodalFoodCalorieModel(
        num_ingredients=num_ingredients,
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")

    # Определение loss function
    criterion = nn.L1Loss()  # MAE (Mean Absolute Error)

    # Определение оптимизатора
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }

    best_mae = float('inf')

    # Обучение
    print(f"Начало обучения на {config['training']['epochs']} эпох...")
    print("=" * 50)

    for epoch in range(config['training']['epochs']):
        print(f"\nЭпоха {epoch + 1}/{config['training']['epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, test_loader, criterion, device)
        scheduler.step(val_mae)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'config': config
            }, config['model']['save_path'])
            print(f"✓ Сохранена лучшая модель (MAE: {val_mae:.4f})")
    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучший Val MAE: {best_mae:.4f}")
    return history, best_mae


def load_model(checkpoint_path, num_ingredients, device='cpu'):
    """
    Загрузка обученной модели
    """
    model = MultimodalFoodCalorieModel(num_ingredients=num_ingredients, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Модель загружена из {checkpoint_path}")
    print(f"Эпоха: {checkpoint['epoch']}")
    print(f"Val MAE: {checkpoint['val_mae']:.4f}")

    return model
