import torch
import torch.nn as nn
import time
import os
import numpy as np
from fully_connected_basics.datasets import get_mnist_loaders
from utils.model_utils import FullyConnectedModel
from utils.experiment_utils import setup_experiment_logging, train_model
from utils.visualization_utils import plot_training_history, count_parameters


def create_width_configs():
    """Создает конфигурации для экспериментов с разной шириной слоев"""
    width_configs = {
        'narrow': [64, 32, 16],  # Узкие слои
        'medium': [256, 128, 64],  # Средние слои
        'wide': [1024, 512, 256],  # Широкие слои
        'very_wide': [2048, 1024, 512]  # Очень широкие слои
    }

    configs = {}
    for name, widths in width_configs.items():
        configs[name] = {
            "input_dim": 784,
            "output_dim": 10,
            "layers": [
                {"type": "linear", "size": widths[0]},
                {"type": "relu"},
                {"type": "linear", "size": widths[1]},
                {"type": "relu"},
                {"type": "linear", "size": widths[2]},
                {"type": "relu"}
            ]
        }

    return configs, width_configs


def run_width_comparison():
    """Запускает сравнение моделей с разной шириной"""
    logger = setup_experiment_logging("results/width_experiments", "width_comparison")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    configs, width_configs = create_width_configs()
    results = {}

    logger.info("Начинаем эксперименты с разной шириной слоев")

    for config_name, config in configs.items():
        logger.info(f"Эксперимент: {config_name} - {width_configs[config_name]}")

        # Создание модели
        model = FullyConnectedModel(config_dict=config)
        param_count = count_parameters(model)

        logger.info(f"Количество параметров: {param_count:,}")

        # Измерение времени обучения
        start_time = time.time()

        # Обучение модели
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            logger=logger,
            epochs=20,
            lr=0.001
        )

        training_time = time.time() - start_time

        # Сохранение результатов
        results[config_name] = {
            'widths': width_configs[config_name],
            'param_count': param_count,
            'training_time': training_time,
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'best_test_acc': max(history['test_accs']),
            'history': history
        }

        # Визуализация истории обучения
        plot_training_history(
            history=history,
            save_path=f"results/width_experiments/{config_name}_history.png",
            title=f"Ширина слоев: {width_configs[config_name]}"
        )

        logger.info(f"Время обучения: {training_time:.2f} сек")
        logger.info(f"Финальная точность: {history['test_accs'][-1]:.4f}")
        logger.info("-" * 50)

    return results


def analyze_width_results(results):
    """Анализирует результаты экспериментов с шириной"""
    print("\n=== АНАЛИЗ ВЛИЯНИЯ ШИРИНЫ СЛОЕВ ===")
    print(f"{'Конфигурация':<12} {'Слои':<20} {'Параметры':<12} {'Время(с)':<10} {'Тест точность':<15}")
    print("-" * 80)

    for config_name, result in results.items():
        widths_str = str(result['widths'])
        param_count = result['param_count']
        training_time = result['training_time']
        test_acc = result['final_test_acc']

        print(f"{config_name:<12} {widths_str:<20} {param_count:<12,} {training_time:<10.1f} {test_acc:<15.4f}")

    # Найти лучшую конфигурацию
    best_config = max(results.items(), key=lambda x: x[1]['final_test_acc'])
    print(f"\nЛучшая конфигурация: {best_config[0]} с точностью {best_config[1]['final_test_acc']:.4f}")

    # Анализ эффективности (точность на параметр)
    print("\n=== АНАЛИЗ ЭФФЕКТИВНОСТИ ===")
    for config_name, result in results.items():
        efficiency = result['final_test_acc'] / (result['param_count'] / 1000)  # точность на 1K параметров
        print(f"{config_name}: {efficiency:.6f} точности на 1K параметров")


def create_architecture_patterns():
    """Создает различные схемы изменения ширины слоев"""
    patterns = {
        # Расширяющиеся архитектуры
        'expanding_small': [128, 256, 512],
        'expanding_medium': [256, 512, 1024],
        'expanding_large': [512, 1024, 2048],

        # Сужающиеся архитектуры
        'narrowing_small': [512, 256, 128],
        'narrowing_medium': [1024, 512, 256],
        'narrowing_large': [2048, 1024, 512],

        # Постоянная ширина
        'constant_small': [256, 256, 256],
        'constant_medium': [512, 512, 512],
        'constant_large': [1024, 1024, 1024],

        # Пирамидальные (сужение к середине)
        'pyramid_small': [512, 128, 512],
        'pyramid_medium': [1024, 256, 1024],
        'pyramid_large': [2048, 512, 2048],

        # Обратная пирамида (расширение к середине)
        'inv_pyramid_small': [128, 512, 128],
        'inv_pyramid_medium': [256, 1024, 256],
        'inv_pyramid_large': [512, 2048, 512]
    }

    configs = {}
    for name, widths in patterns.items():
        configs[name] = {
            "input_dim": 784,
            "output_dim": 10,
            "layers": [
                {"type": "linear", "size": widths[0]},
                {"type": "relu"},
                {"type": "linear", "size": widths[1]},
                {"type": "relu"},
                {"type": "linear", "size": widths[2]},
                {"type": "relu"}
            ]
        }

    return configs, patterns


def run_grid_search():
    """Выполняет grid search для поиска оптимальной архитектуры"""
    logger = setup_experiment_logging("results/width_experiments", "grid_search")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    configs, patterns = create_architecture_patterns()
    grid_results = {}

    logger.info("Начинаем grid search для оптимизации архитектуры")

    for config_name, config in configs.items():
        logger.info(f"Тестируем архитектуру: {config_name} - {patterns[config_name]}")

        model = FullyConnectedModel(config_dict=config)
        param_count = count_parameters(model)

        # Обучение с меньшим количеством эпох для grid search
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            logger=logger,
            epochs=15,
            lr=0.001
        )

        grid_results[config_name] = {
            'pattern': patterns[config_name],
            'param_count': param_count,
            'final_test_acc': history['test_accs'][-1],
            'best_test_acc': max(history['test_accs']),
            'convergence_epoch': np.argmax(history['test_accs']) + 1,
            'history': history
        }

        logger.info(f"Точность: {history['test_accs'][-1]:.4f}, Параметры: {param_count:,}")

    return grid_results


def create_results_heatmap(grid_results):
    """Создает heatmap результатов grid search"""
    from utils.visualization_utils import plot_heatmap

    # Группировка результатов по типам архитектур
    pattern_types = ['expanding', 'narrowing', 'constant', 'pyramid', 'inv_pyramid']
    sizes = ['small', 'medium', 'large']

    # Создание матрицы результатов
    heatmap_data = np.zeros((len(pattern_types), len(sizes)))
    labels_x = sizes
    labels_y = pattern_types

    for i, pattern_type in enumerate(pattern_types):
        for j, size in enumerate(sizes):
            key = f"{pattern_type}_{size}"
            if key in grid_results:
                heatmap_data[i, j] = grid_results[key]['final_test_acc']

    # Создание heatmap
    plot_heatmap(
        data=heatmap_data,
        xlabels=labels_x,
        ylabels=labels_y,
        save_path="results/width_experiments/architecture_heatmap.png",
        title="Точность классификации для различных архитектур"
    )

    print("Heatmap сохранена в results/width_experiments/architecture_heatmap.png")


def analyze_grid_search_results(grid_results):
    """Анализирует результаты grid search"""
    print("\n=== РЕЗУЛЬТАТЫ GRID SEARCH ===")

    # Сортировка по точности
    sorted_results = sorted(grid_results.items(), key=lambda x: x[1]['final_test_acc'], reverse=True)

    print(f"{'Архитектура':<20} {'Схема':<20} {'Параметры':<12} {'Точность':<12} {'Сходимость':<12}")
    print("-" * 90)

    for config_name, result in sorted_results[:10]:  # Топ-10 результатов
        pattern_str = str(result['pattern'])
        param_count = result['param_count']
        accuracy = result['final_test_acc']
        convergence = result['convergence_epoch']

        print(f"{config_name:<20} {pattern_str:<20} {param_count:<12,} {accuracy:<12.4f} {convergence:<12}")

    # Анализ по типам архитектур
    print("\n=== АНАЛИЗ ПО ТИПАМ АРХИТЕКТУР ===")
    pattern_analysis = {}

    for config_name, result in grid_results.items():
        pattern_type = config_name.split('_')[0] + '_' + config_name.split('_')[1]
        if pattern_type not in pattern_analysis:
            pattern_analysis[pattern_type] = []
        pattern_analysis[pattern_type].append(result['final_test_acc'])

    for pattern_type, accuracies in pattern_analysis.items():
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{pattern_type:<20}: {avg_acc:.4f} ± {std_acc:.4f}")

    # Лучшая архитектура
    best_arch = sorted_results[0]
    print(f"\nЛУЧШАЯ АРХИТЕКТУРА: {best_arch[0]}")
    print(f"Схема: {best_arch[1]['pattern']}")
    print(f"Точность: {best_arch[1]['final_test_acc']:.4f}")
    print(f"Параметры: {best_arch[1]['param_count']:,}")


def main():

    # Создание директорий
    os.makedirs("results/width_experiments", exist_ok=True)

    print("=== ЭКСПЕРИМЕНТЫ С ШИРИНОЙ СЕТИ ===")

    # 2.1 Сравнение моделей разной ширины
    print("\n1. Запуск сравнения моделей разной ширины...")
    width_results = run_width_comparison()
    analyze_width_results(width_results)

    # 2.2 Grid search оптимизация
    print("\n2. Запуск grid search для оптимизации архитектуры...")
    grid_results = run_grid_search()
    analyze_grid_search_results(grid_results)

    # Создание heatmap
    print("\n3. Создание heatmap результатов...")
    create_results_heatmap(grid_results)

    print("\n=== ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ ===")
    print("Результаты сохранены в results/width_experiments/")


if __name__ == "__main__":
    main()
