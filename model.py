import torch
import torch.nn as nn

# Ручная архитектура InceptionTime
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super().__init__()
        
        # Bottleneck 1x1 уменьшает число каналов перед свёртками
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            in_channels = bottleneck_channels
            
        # Несколько свёрток с разными размерами ядра позволяют искать признаки на разных временных масштабах ЭКГ-сигнала
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2, bias=False))

        # Ветка с max pooling помогает учитывать локальные максимумы сигнала
        self.maxconv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        # После объединения всех веток: три свёрточные ветки + одна ветка max pooling
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        # Если используется bottleneck, сначала пропускаем сигнал через 1x1 свёртку
        if self.use_bottleneck:
            input_tensor = self.bottleneck(input_tensor)
        # Применяем все параллельные ветки Inception-блока
        outputs = [conv(input_tensor) for conv in self.convs]
        outputs.append(self.maxconv(input_tensor))
        # Объединяем признаки, полученные разными ветками, по каналам
        x = torch.cat(outputs, dim=1)
        return self.act(self.bn(x))

# Shortcut нужен для остаточной связи между блоками. Уменьшает проблему затухания градиентов
class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Свёртка 1x1 приводит число каналов к нужному размеру,чтобы можно было сложить shortcut и выход Inception-блоков
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

# Основная модель InceptionTime для классификации 12-канальной ЭКГ
class InceptionTimeECG(nn.Module):
    def __init__(self, c_in=12, c_out=5, depth=6, nb_filters=32):
        super().__init__()
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        # curr_in — текущее число входных каналов,
        # curr_out — число каналов после одного Inception-блока
        curr_in, curr_out = c_in, nb_filters * 4
        for d in range(depth):
            self.blocks.append(InceptionModule(curr_in, nb_filters))
            # Остаточная связь добавляется после каждого третьего Inception-блока
            if d % 3 == 2:
                s_in = c_in if d == 2 else curr_out
                self.shortcuts.append(Shortcut(s_in, curr_out))
            else:
                self.shortcuts.append(None)
            curr_in = curr_out
        # Global Average Pooling переводит временную последовательность в один вектор признаков для каждого объекта
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Финальный полносвязный слой выдаёт 5 логитов, по одному для каждого диагностического суперкласса
        self.fc = nn.Linear(curr_out, c_out)

    def forward(self, x):
        res = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Если для текущего блока задан shortcut, добавляем остаточную связь
            if self.shortcuts[i] is not None:
                res = self.shortcuts[i](res)
                x = torch.relu(x + res)
                res = x
        return self.fc(self.gap(x).squeeze(-1))
