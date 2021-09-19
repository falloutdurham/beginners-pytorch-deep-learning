# PyTorch for Deep Learning: Creating and Deploying Deep Learning Applications

Repository for scripts and notebooks from the book: Programming PyTorch for Deep Learning: Creating and Deploying Deep Learning Applications

## Download of dataset for chapter 2 ([download.py](https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter2/download.py))

Since some links are broken meanwhile, you can also find a downloadable version of the image dataset here (zip file): https://drive.google.com/file/d/16h8E7dnj5TpxF_ex4vF2do20iMWziM70

## Updates

* 2020/05/25: Chapter 9.75 — Image Self-Supervised Learning

* 2020/03/01: Chapter 9.5 - Text Generation With GPT-2 And (only) PyTorch, or Semi/Self-Supervision Learning Part 1 (Letters To Charlotte)

* 2020/05/03: Chapter 7.5 - Quantizing Models 

________________

# Deutschsprachige Ausgabe
## PyTorch für Deep Learning: Anwendungen für Bild-, Ton- und Textdaten entwickeln und deployen

--> [https://dpunkt.de/produkt/pytorch-fuer-deep-learning/]()

## Hinweis zum Download des Datensatzes in Kapitel 2 ([download.py](https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter2/download.py))

Da einige URLs inzwischen leider veraltet sind, stehen Ihnen die Bilddateien zusätzlich als Download (Zip-Datei) bereit: https://drive.google.com/file/d/16h8E7dnj5TpxF_ex4vF2do20iMWziM70 

## Installationshinweise

 - [Python - Downloads und Dokumentation](https://www.python.org/)
 - [Anaconda - Dokumentation mit Installationshinweisen](https://docs.anaconda.com/anaconda/)
 - [pip - Installationshinweise](https://pypi.org/project/pip/)
 - [PyTorch - Installationshinweise](https://pytorch.org/get-started/locally/)
   + falls Installation nicht mit `conda env create --file environment.yml`/`pip3 install -r requirements.txt
   /requirements_cuda_available.txt` erfolgt; ansonsten siehe Abschnitt [_Versionskontrolle_](#Versionskontrolle)
 - [Jupyter Notebook / JupyterLab - Installation und Dokumentation](https://jupyter.org/)
 - [Google Colaboratory - Einführung und weitergehende Hinweise insb. zum Einlesen von Daten](https://colab.research.google.com/notebooks/intro.ipynb)
 - [Github - Forken und Klonen eines Repositorys](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo)
 
## Versionskontrolle

Nachdem Sie das Github-Repository lokal geklont (bzw. zuvor geforkt) haben!

### Conda

1.) Wechseln Sie zunächst in den Zielordner (`cd beginners-pytorch-deep-learning`), erstellen Sie dann eine (lokale) virtuelle Umgebung und installieren Sie die benötigten Bibliotheken und Pakete:

`conda env create --file environment.yml`

2.) Anschließend aktivieren Sie die virtuelle Umgebung:

`conda activate myenv`

3.) Zum Deaktivieren nutzen Sie den Befehl:

`conda deactivate`

### pip

1.) Wechseln Sie zunächst in den Zielordner (`cd beginners-pytorch-deep-learning`) und erstellen Sie anschließend eine
 virtuelle Umgebung:

`python3 -m venv myenv`

2.) Aktivieren Sie die virtuelle Umgebung (https://docs.python.org/3/library/venv.html):

`source myenv/bin/activate` (Ubuntu/Mac)
`myenv\Scripts\activate.bat` (Windows)
 
3.) Erstellen Sie eine (lokale) virtuelle Umgebung und installieren Sie die benötigten Bibliotheken und Pakete:

`pip3 install -r requirements.txt`


4.) Zum Deaktivieren nutzen Sie den Befehl:

`deactivate`

### Bei Nutzung von Jupyter Notebook

1.) Zunächst müssen Sie Jupyter Notebook installieren:

 `conda install -c conda-forge notebook` oder `pip3 install notebook`

2.) Nach Aktivierung Ihrer virtuellen Umgebung (s.o.) geben Sie den folgenden Befehl in Ihre Kommandozeile ein, um die
 `ipykernel`-Bibliothek herunterzuladen:
 
 `conda install ipykernel` oder `pip3 install ipykernel`
 
3.) Installieren Sie einen Kernel mit Ihrer virtuellen Umgebung:

 `ipython kernel install --user --name=myenv`

4.) Starten Sie Jupyter Notebook:

 `jupyter notebook`
 
5.) Nach Öffnen des Jupyter-Notebook-Startbildschirms wählen Sie auf der rechten Seite das Feld _New_ (bzw. in der
 Notebook-Ansischt den Reiter _Kernel_/_Change Kernel_) und wählen Sie _myenv_ aus.
 
### Google Colaboratory

Hier stehen Ihnen hier für mehrere Stunden leistungsfähige GPUs zur Verfügung, die das Training der Modelle merklich beschleunigen können. In Google Colab stehen Ihnen standardmäßig einige Pakete bereits vorinstalliert zur Verfügung. Da sich
 Neuinstallationen immer nur auf ein Notebook beziehen, können Sie von einer Einrichtung einer virtuellen Umgebung
  absehen und direkt die Pakete durch Ausführen der Zellen bzw. Zeilen, in denen ein **!** vorangestellt ist, installieren.
