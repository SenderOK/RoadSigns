Распознавание дорожных знаков с помощью метода опорных векторов.

Работу выполнил Сендерович Никита Леонидович, гр. 317.

Дополнительные части:
1) Повышенное качество классификации (поживём --- увидим)
2) Нелинейное ядро chi-square для SVM

Особенности реализации:
1) Для построения вектора признаков по каждой из картинок с определённым шагом 
происходит пробег большим блоком, составленным из 9 клеток. Для каждой из клеток
блока считается гистограмма, они конкатенируются, нормализуются и добавляются в
вектор признаков. Шаг выбирается так, чтобы блоки накладывались в соответствии
с заданным параметром OVERLAP.

2) Для нормализации используется рекомендованная Далалом и Триггсом норма L2Hys,
которая отличается от обычной Евклидовой нормы отсечением значений в гистограмме
по порогу 0.2 с последующей ренормализацией.
