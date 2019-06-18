#%%
import json
import seaborn as sns
sns.set()
import os
#Curve format {"line":{"color":"navy","dash":"dash","width":2},"mode":"lines","showlegend":false,"type":"scatter","uid":"bb0382a9-56cd-418f-b266-5ce7255d3a09","x":[0,1],"y":[0,1]}


def readAvgCurveFromJson(filename: str, directory: str) -> dict:
    path = os.path.join(directory, filename)
    with open(path) as f:
        data = json.loads(f.read())
    if len(data) > 1:
        avgCurveData = data[1]
    else:
        avgCurveData = data[0]
    return avgCurveData


def plotRocCurves(curveList, labelsList, title, xlim=None, ylim=None):
    import matplotlib.pyplot as plt
    plt.figure(dpi=300)
    for (curveData, label) in zip(curveList, labelsList):
        plt.plot(curveData['x'], curveData['y'], label=label)
        plt.title(title)
        plt.legend()
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.show()


def getTitleFormFileName(fileName: str) -> str:
    experiment = fileName.split('-')[0][1]
    realisation = fileName.split('-')[1][1]
    return 'Eksperyment {}, realizacja {}'.format(experiment, realisation)


def getLabelFromFileName(fileName: str) -> str:
    splt = fileName.split('-')
    if len(splt) > 3:
        return 'Epoka {} - średnia krzywa ROC'.format(splt[2])
    else:
        return 'Zbiór testowy - średnia krzywa ROC'


def processFiles(files, title='', labels=[], xlim=None, ylim=None):
    curves = [readAvgCurveFromJson(file, directory='roc_curve_data') for file in files]
    plotRocCurves(curves, labelsList=labels if len(labels) else
    [getLabelFromFileName(file) for file in files],
        title=title if title else getTitleFormFileName(files[0]), xlim=xlim, ylim=ylim)


if __name__ == "__main__":

    # %%
    files_e1_r1 = ['e1-r1-0-val.html', 'e1-r1-5-val.html', 'e1-r1-45-val.html', 'e1-r1-test.html']
    processFiles(files_e1_r1)
    #%%
    files_e1_r2 = ['e1-r2-0-val.html', 'e1-r2-5-val.html', 'e1-r2-45-val.html', 'e1-r2-test.html']
    processFiles(files_e1_r2)
    #%%
    files_e1_r1_r2 = ['e1-r1-45-val.html', 'e1-r1-test.html', 'e1-r2-45-val.html', 'e1-r2-test.html', 'e1-r3-65-val.html','e1-r3-test.html' ]
    processFiles(files_e1_r1_r2, title='Eksperyment 1',
                 xlim=[0,0.5], ylim=[0.7,1],labels=[
        'Realizacja 1, Epoka 45 - średnia krzywa ROC',
        'Realizacja 1, Zbiór testowy - średnia krzywa ROC',
        'Realizacja 2, Epoka 45 - średnia krzywa ROC',
        'Realizacja 2, Zbiór testowy - średnia krzywa ROC',
        'Realizacja 3, Epoka 65 - średnia krzywa ROC',
        'Realizacja 3, Zbiór testowy - średnia krzywa ROC',
    ])
    #%%
    files_e2_r1 = ['e2-r1-0-val.html', 'e2-r1-5-val.html', 'e2-r1-45-val.html', 'e2-r1-test.html']
    processFiles(files_e2_r1)
    #%%
    files_e2_r2 = ['e2-r2-0-val.html', 'e2-r2-5-val.html', 'e2-r2-45-val.html', 'e2-r2-test.html']
    processFiles(files_e2_r2)
    #%%
    files_e1_e2 = ['e1-r1-test.html', 'e1-r2-test.html', 'e2-r1-test.html', 'e2-r2-test.html']
    processFiles(files_e1_e2, title='Eksperyment 1 vs. Eksperyment 2 - zbiory testowe',
                 labels=[getTitleFormFileName(file) for file in files_e1_e2])
    # %%
    files_e2_r1_r2 = ['e2-r1-45-val.html', 'e2-r1-test.html', 'e2-r2-45-val.html', 'e2-r2-test.html',
                      'e2-r3-65-val.html', 'e2-r3-test.html']
    processFiles(files_e2_r1_r2, title='Eksperyment 2',
                 xlim=[0, 0.5], ylim=[0.7, 1], labels=[
            'Realizacja 1, Epoka 45 - średnia krzywa ROC',
            'Realizacja 1, Zbiór testowy - średnia krzywa ROC',
            'Realizacja 2, Epoka 45 - średnia krzywa ROC',
            'Realizacja 2, Zbiór testowy - średnia krzywa ROC',
            'Realizacja 3, Epoka 65 - średnia krzywa ROC',
            'Realizacja 3, Zbiór testowy - średnia krzywa ROC',
        ])
    # %%
    files_e3_r1_r2A = ['e3-r1-first-45-val.html', 'e3-r1-first-test.html',
                      'e3-r3-first-10-val.html', 'e3-r3-first-test.html']
    processFiles(files_e3_r1_r2A, title='Eksperyment 3A',
                 xlim=[0, 0.5], ylim=[0.7, 1], labels=[
            'Realizacja 1, Epoka 45 - średnia krzywa ROC',
            'Realizacja 1, Zbiór testowy - średnia krzywa ROC',
            'Realizacja 3, Epoka 10 - średnia krzywa ROC',
            'Realizacja 3, Zbiór testowy - średnia krzywa ROC',
        ])
    # %%
    files_e3_r1_r2B = ['e3-r1-second-45-val.html','e3-r1-test.html',
                      'e3-r3-second-10-val.html', 'e3-r3-second-test.html']
    processFiles(files_e3_r1_r2B, title='Eksperyment 3B',
                 xlim=[0, 0.5], ylim=[0.7, 1], labels=[
            'Realizacja 1, Epoka 45 - średnia krzywa ROC',
            'Realizacja 1, Zbiór testowy - średnia krzywa ROC',
            'Realizacja 3, Epoka 10 - średnia krzywa ROC',
            'Realizacja 3, Zbiór testowy - średnia krzywa ROC',
        ])
    #%%
    files_e3_r1_1 = ['e3-r1-first-0-val.html', 'e3-r1-first-45-val.html',
                     'e3-r1-second-0-val .html', 'e3-r1-second-45-val.html',
                     'e3-r1-test.html']
    processFiles(files_e3_r1_1, labels=[
        'Epoka 0A - średnia krzywa ROC',
        'Epoka 45A - średnia krzywa ROC',
        'Epoka 0B - średnia krzywa ROC',
        'Epoka 45B - średnia krzywa ROC',
        'Zbiór testowy - średnia krzywa ROC'
    ])

    #%%
    files_e3_r2_1 = ['e3-r2-first-0-val.html', 'e3-r2-first-65-val.html', 'e3-r2-second-0-val.html',
                     'e3-r2-second-65-val.html', 'e3-r2-second-test.html']
    processFiles(files_e3_r2_1, labels=[
        'Epoka 0A - średnia krzywa ROC',
        'Epoka 65A - średnia krzywa ROC',
        'Epoka 0B - średnia krzywa ROC',
        'Epoka 65B - średnia krzywa ROC',
        'Zbiór testowy - średnia krzywa ROC'

    ])
    # %%
    files_e4_fst = [
        'e4-r1-first_cnn_codes-0-test.html',
        'e4-r1-first_cnn_codes-1-test.html',
        'e4-r1-first_cnn_codes-2-test.html',
        'e4-r3-first_cnn_codes-0-test.html',
        'e4-r3-first_cnn_codes-1-test.html',
        'e4-r3-first_cnn_codes-2-test.html',
    ]
    processFiles(files_e4_fst, title='Eksperyment 4', xlim=[0, 1], ylim=[0.8, 1], labels=[
        'Realizacja 1A, Jądro kwadratowe - średnia krzywa ROC',
        'Realizacja 1A, Jądro liniowe - średnia krzywa ROC',
        'Realizacja 1A, Jądro wielomianowe - średnia krzywa ROC',
        'Realizacja 3A, Jądro kwadratowe - średnia krzywa ROC',
        'Realizacja 3A, Jądro liniowe - średnia krzywa ROC',
        'Realizacja 3A, Jądro wielomianowe - średnia krzywa ROC',
    ])
 # %%
    files_e4_sec = [
        'e4-r1-second_cnn_codes-0-test.html',
        'e4-r1-second_cnn_codes-1-test.html',
        'e4-r1-second_cnn_codes-2-test.html',
        'e4-r3-second_cnn_codes-0-test.html',
        'e4-r3-second_cnn_codes-1-test.html',
        'e4-r3-second_cnn_codes-2-test.html',
    ]
    processFiles(files_e4_sec,title='Eksperyment 4', xlim=[0,0.5], ylim=[0.8,1], labels=[
        'Realizacja 1B, Jądro kwadratowe - średnia krzywa ROC',
        'Realizacja 1B, Jądro liniowe - średnia krzywa ROC',
        'Realizacja 1B, Jądro wielomianowe - średnia krzywa ROC',
        'Realizacja 3B, Jądro kwadratowe - średnia krzywa ROC',
        'Realizacja 3B, Jądro liniowe - średnia krzywa ROC',
        'Realizacja 3B, Jądro wielomianowe - średnia krzywa ROC',
    ])


#%%Best 1-R3, 2-R3, 3A-R3, 3B-R3,4F - 3B1, 4S-31
best_files = ['e1-r3-test.html','e2-r3-test.html', 'e3-r3-first-test.html','e3-r3-second-test.html',
              'e4-r3-second_cnn_codes-1-test.html', 'e4-r3-first_cnn_codes-1-test.html']
processFiles(best_files,title='Najlepsze wyniki - krzywe ROC', xlim=[0,0.4], ylim=[0.9,1], labels=[
        'Eksperyment 1, realizacja 3 - średnia krzywa ROC',
        'Eksperyment 2, realizacja 3 - średnia krzywa ROC',
        'Eksperyment 3A, realizacja 3 - średnia krzywa ROC',
        'Eksperyment 3B, realizacja 3 - średnia krzywa ROC',
        'Eksperyment 4A, realizacja 3 - średnia krzywa ROC',
        'Eksperyment 4B, realizacja 3 - średnia krzywa ROC',
    ])
