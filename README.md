# Edukacyjny Klasyfikator Galaktyk 

## Cel Projektu
Projekt stanowi edukacyjną implementację Konwolucyjnej Sieci Neuronowej zbudowanej od podstaw przy użyciu biblioteki TensorFlow/Keras. 
Głównym zadaniem modelu jest analiza obrazów astronomicznych i automatyczne rozróżnianie morfologii galaktyk spiralnych i eliptycznych.

## Zbiór i Przetwarzanie Danych


* Filtrowanie: Skrypt pobiera wyłącznie zdjęcia o bardzo wysokim współczynniku pewności ludzkiej klasyfikacji - powyżej 80%.
* Balansowanie próby: Do nauki wyselekcjonowano równy zbiór po 500 najlepszych reprezentantów dla każdej z klas, aby zapobiec faworyzowaniu jednej z nich przez model.
* Data Augmentation: Aby zniwelować zjawisko overfittingu, zaobserwowane w bazowych iteracjach modelu, zaimplementowano dynamiczną augmentację.
  Obrazy treningowe są w locie poddawane losowym obrotom, przesunięciom, ścinaniu, przybliżaniu oraz odbiciom lustrzanym.

## Architektura Modelu (v3)
Klasyfikator wykorzystuje klasyczną, sekwencyjną architekturę ekstrakcji cech:
1. Bloki Konwolucyjne: Trzy pary warstw `Conv2D` + `MaxPooling2D` o rosnącej liczbie filtrów (32 ➔ 64 ➔ 128), przetwarzające wejściowe obrazy (150x150 px).
2. Spłaszczenie: Warstwa `Flatten` transformująca macierze do postaci jednowymiarowej wektora.
3. Mózg Decyzyjny: W pełni połączona warstwa ukryta (`Dense`) z 512 neuronami.
4. Warstwa Wyjściowa: Dwa neurony z funkcją aktywacji `Softmax`, zwracające ostateczne prawdopodobieństwo procentowe dla obu klas.

Model jest kompilowany z optymalizatorem `Adam` i zapisywany na dysku jako `galaxy_classifier.keras`.

