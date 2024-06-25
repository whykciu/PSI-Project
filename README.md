# PSI-Project
## Model przewidujący przyszłe ceny akcji
Poniższy projekt ma na celu predykcję cen akcji na giełdzie, a w zasadzie przewidywanie trendu, czy cena w przyszłości wzrośnie czy też spadnie. Końcowo sprowadza się to do problemu klasyfikacji danych do jednej z dwóch klas:
* Cena akcji w następnym dniu wzrośnie - oznaczane przez 1
* Cena akcji w następnym dniu spadnie - oznaczane przez 0

## Dane wejściowe
Dane użyte do predykcji pochodzą z Yahoo Finance za pomocą biblioteki `yfinance`. W moim przypadku korzystam z cen akcji firmy Apple. Początkowe dane przechodzą prosty preprocessing aby uzyskać najkorzystniejsze wyniki. Z danych usunięte zostają kolumny `Stock Splits` i `Dividends`. W celu predykcji dodana zostaje kolumna `Tomorrow`, przechowującą wartość zamknięcia następnego dnia. Z porównania tej wartości do wartość dnia aktualnego utworzona zostaje dodatkowa kolumna `Target`, która jest wartością, którą chcemy przewidzieć. Wartości w tej kolumnie są przypisywane na zasadzie podziału klasyfikacji klas podanych we wstępie. Ostatecznie początkowe atrybuty danych to:
* **Open**: Cena otwarcia
* **High**: Najwyższa cena dnia
* **Low**: Najniższa cena dnia
* **Close**: Cena zamknięcia
* **Volume**: Wolumen obrotu
* **Tomorrow**
* **Target**  

Dodatkowo w przypadku cen akcji na giełdzie i ze względu na zmiany na rynku, gdzie stare dane mogą nie mieć wpływu na aktualne dane,  optymalnie jest nie wykorzystywać całej historii cen akcji, dlatego brane pod uwagę są tylko dane od 1996 roku.  

## Dodatkowe predyktory 
Początkowe predyktory to większość podstawowych atrybutów: `"Open", "High", "Low", "Close", "Volume"`. W celu poprawy dokładności i niezawodności modelu dodane zostały dodatkowe predyktory wykorzystujące okna czasowe (rolling window), czyli pewne okresy czasu na których przeprowadzone zostały wyliczenia nowych predyktorów. Model wykorzystuje te dane do lepszej i głębszej analizy trendów i wartości cen w przeszłości. Dodane nowe predyktory to: 
* **Close ratio** - stosunek ceny zamknięcia akcji do jej średniej kroczącej z określonego okresu (okna czasowego)
* **Trend** - sumuje wartości celu (czyli czy cena wzrosła czy spadła) z poprzednich dni w danym oknie czasowym
* **RSI** - (Relative Strength Index) indeks siły względnej, mierzy prędkość i zmianę ruchów cenowych w danym oknie czasowym
* **SMA** - (Simple Moving Average) prosta średnia krocząca (średnia cen z kolumny `Close`) z określonego okna czasowego
* **EMA** - (Exponential Moving Average) wykładnicza średnia krocząca, z określonego okna czasowego, która nadaje większą wagę nowszym danym i szybciej reaguje na najnowsze zmiany cen
* **MACD** - (Moving Average Convergence Divergence) jest różnicą między krótkoterminową a długoterminową EMA, która pomaga zidentyfikować zmiany w kierunku, sile, momentum i długości trendu akcji 
* **Signal MACD** - to wartość EMA z MACD

## Strategia podziału danych
Na podstawie metody prób i błędów, testując podziały: 80/20, 85/15, 90/10 model uzyskuje najlepsze wyniki przy podziale **80% dane treningowe**, **20% dane testowe**. Przy czym w przykładzie predykcji opierającej się na danych w przestrzeni czasowej musimy pamiętać aby podział danych testowych i treningowych odbywał się **zgodnie z chronologią**, zatem dane testowe muszą być danymi które są chronologicznie po danych treningowych.

## Wybór modelu
W projekcie przetestowane zostały trzy popularne klasyfikatory: 
* **AdaBoostClassifier** - iteracyjnie wzmacnia słabe klasyfikatory, co prowadzi do tworzenia bardziej precyzyjnego modelu.
* **XGBClassifier** - zapewnia lepszą kontrolę nad nadmiernym dopasowaniem poprzez użycie regularyzacji L1 i L2
* **RandomForestClassifier** - Dzięki tworzeniu wielu drzew decyzyjnych i ich agregacji, Random Forest redukuje wariancję w prognozach, co prowadzi do bardziej stabilnych wyników, unika nadmiernego dopasowania modelu do danych
* **VotingClassifier** - komitet klasyfikatorów złożony z wcześniej wymienionych, pozwala na maksymalizację indywidualnych zalet każdego z modeli i uzyskanie lepszych wyników

W celu dobrania jak najlepszych parametrów do tych modeli użyłem techniki GridSearchCV, gdzie jako cross-validation używam **TimeSeriesSplit**. Parametry zostały dobrane tak aby zmaksymalizować wynik **precision**. W modelu predykcji cen akcji wydaje się to być najrozsądniejszą opcją, gdyż wartości predykcji **false positive** są znacznie bardziej szkodliwe niż **false negative**. Wiąże się to z tym, że dla **FP** możemy fizycznie stracić część inwestycji, natomiast dla **FN** możemy po prostu nie zyskać. Dlatego też bardziej opłacalne jest maksymalizowanie wartości **precision** aniżeli **recall**, aby mieć większą pewność, że wartość akcji wzrośnie wtedy, gdy model tak przewiduje.

## Wyniki
Dla najlepszych parametrów wybranych przez metodę GridSearch
* **AdaBoost Classifier**
    * Precision: ~ 0.50 - 0.55
    * Accuracy: ~ 0.50 - 0.55
    * Recall: ~ 0.95 - 1.00
    * F1: ~ 0.70
* **Random Forest Classifier**
    * Precision: ~ 0.50 - 0.55
    * Accuracy: ~ 0.50
    * Recall: ~ 0.10 - 0.15
    * F1: ~ 0.20
* **XGBoost Classifier**
    * Precision: ~ 0.50 - 0.55
    * Accuracy: ~ 0.50
    * Recall: ~ 0.70 - 0.75
    * F1: ~ 0.60
* **Voting Classifier (Ensemble)**
    * Precision: ~ 0.55
    * Accuracy: ~ 0.50 - 0.55
    * Recall: ~ 0.45
    * F1: ~ 0.50

Jak możemy zauważyć wyniki w kryterium **precision** są dość do siebie zbliżone i zakrawają o prawdopodobieństwo rzutu monetą. Wyniki **recall** natomiast są zaskakująco wysokie dla AdaBoost oraz XBG. Modele te często poprawnie przewidują wzrost cen wtedy gdy cena faktycznie wzrasta. Jednakże nie biorą pod uwagę ile było false positive predykcji. Warto dodać, że dla niektórych parametrów, nie wybranych przez grid search, modele (w szczególności AdaBoost, gdzie dochodziło do wyniiku 0.70) otrzymywały lepsze wyniki w **precision** a gorsze w **recall** (przeważnie przy zwiększaniu parametru n_estimators). Najbardziej zrównoważone wyniki otrzymujemy w VotingClassifier. Tam też wynik **precision** jest największy. Podsumowując wyniki nie są zadowalające, jednakże wciąż możemy probować zmieniać parametry, tak aby **precision** było jak najlepsze, dodatkowo dodając nowe predyktory.

## Propozycje dalszych kroków
Model ten może mieć znacznie bardziej zadowalające wyniki i aby to uzyskać można wykorzystać poniższe propozycje:
* Dalszą optymalizację parametrów.
* Eksperymentowanie z dodatkowymi wskaźnikami technicznymi, dodając nowe indeksy, sentymenty danej spółki w danym okresie czasu, dniu.
* Integrację z systemami transakcyjnymi.

#### Wymagania
* Python 3.7+
* Jupyter Notebook (opcjonalne)
* Biblioteki:
    * yfinance
    * pandas
    * scikit-learn
    * xgboost
    * joblib

#### Instalacja i uruchomienie
Instalacja wymaganych bibliotek:
```
pip install yfinance pandas scikit-learn xgboost
```

Uruchomienie pliku Jupyter, aby wykonać wszystkie etapy uczenia i ewaluacji modelu. Początkowo przejdź do folderu z plikiem `.ipynb`. Następnie wpisz komendę:
```
jupyter notebook
```   
Która przeniesie cię do przeglądarki z gotowym plikiem.

Inna opcja to uruchomienie samego pliku Python. Początkowo przejdź do folderu z plikiem `.py`
Aby uruchomić naukę i zapis modeli wpisz komendę:
```
python stock_market_pred.py
```   

Aby wykonać predykcję dla zapisanych modeli uruchom:
```
python predict_latest.py
```  
#### Dziękuje za poświęcony czas!

#### Autor projektu: Hubert Buś