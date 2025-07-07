![1.1 Сравнение на MNIST](https://github.com/petalsun/homework_ogo/blob/homework4/plots/final1.png)


![DeepFCN_Matrix](https://github.com/petalsun/homework_ogo/blob/homework4/plots/DeepFcnMatr.png)
![DeepFCN](https://github.com/petalsun/homework_ogo/blob/homework4/plots/deepfcn.png)
![ResNetCIFAR_Matrix](https://github.com/petalsun/homework_ogo/blob/homework4/plots/ResNetCifarMatr.png)
![ResNetCIFAR](https://github.com/petalsun/homework_ogo/blob/homework4/plots/ResNetCifar.png)
![RegulResNet_Matrix](https://github.com/petalsun/homework_ogo/blob/homework4/plots/ReguResNetMatr.png)
![RegulResNet](https://github.com/petalsun/homework_ogo/blob/homework4/plots/ReguResNet.png)

![1.2 Сравнение на CIFAR-10](https://github.com/petalsun/homework_ogo/blob/homework4/plots/final2.png)

Сравнительный анализ архитектур нейронных сетей показал существенные различия в их эффективности для обработки изображений. Полносвязная сеть (DeepFCN) продемонстрировала неудовлетворительные результаты, что подтверждает ее слабую применимость для задач компьютерного зрения. Базовая версия ResNet, несмотря на высокую точность на обучающей выборке, показала признаки значительного переобучения. Введение методов регуляризации позволило создать более сбалансированную модель (RegularizedResNet), где удалось минимизировать разрыв между обучающей и тестовой производительностью, сохранив при этом приемлемый уровень точности. Результаты убедительно свидетельствуют о преимуществе ResNet-архитектур перед полносвязными сетями для работы с изображениями.

