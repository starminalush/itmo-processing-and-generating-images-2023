Репозиторий с домашними заданиями для курса "Обработка и генерация изображений"

Изюмова Анастасия Витальевна

## Задача
Задача: Детекция людей с камеры дронов

Датасет: [Lacmus Drone Dataset (LADD)](https://www.kaggle.com/datasets/mersico/lacmus-drone-dataset-ladd-v40)

Классы: ['people']

## Эксперимент:
1. Получить бейзлайн детекции людей

## Модель
Архитектура: YOLOv5
Базовая модель: yolov5s
### Гиперпараметры
- optimizer: SGD
- lr0: 0.01
- momentum: 0.937
- num_epochs: 15
- batch: 2
- imgsz: 1280
### Лоссы:
#### Train
![image](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/f44d0e54-dedb-499b-8687-31a927145fff)
![image](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/521aaf9d-9b39-4e2c-9237-93e023114eee)
![image](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/f37690b1-1b48-4fe3-a57c-98a5feb83e19)
#### Validation
![image](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/dab2b3be-26b7-488f-8105-6590b1b6a374)
![image](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/0c2def5b-2e36-4394-b096-b4cd0ae5c0a7)
![image](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/251ac0ca-9564-4e96-a04c-272d0d45d2ce)

### Метрики
Так как у меня всего один класс, то метрики по всем классам = метрики по классу people
|Precision|Recall|F1|MAP50|MAP50-95|
|---|---|---|---|---|
|0.88|0.85|0.87|0.89|0.53|

![F1_curve](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/500c32fc-da57-46d3-b42c-ceaa001f8026)
![P_curve](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/ad263a33-3848-4fa6-9c68-d88f979a12d3)
![PR_curve](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/af60f79c-ee57-45d2-a46b-b902dfb580a7)
![R_curve](https://github.com/starminalush/itmo-processing-and-generating-images-2023/assets/103132748/66553952-c8ad-4705-8d12-46843cbe721a)

## Выводы:
Текущее качество модели устраивает, считаем бейзлан полученным
