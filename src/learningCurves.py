#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
from typing import List

#%%
def plotLearningCurves(curveList :List[pd.DataFrame], labelsList, title, ylim=None):
    plt.figure(dpi=300)
    for (curveData, label) in zip(curveList, labelsList):
        plt.plot(curveData['Step'], curveData['Value'], label=label)
        plt.title(title)
        plt.legend(prop={'size': 8})
        if ylim:
            plt.ylim(ylim)
    plt.show()

def readCurveFromFile(filename, directory):
    path = os.path.join(directory, filename)
    return pd.read_csv(path)

def processFiles(files,directory, title='', labels=[], ylim=None):
    curves = [readCurveFromFile(file, directory=directory) for file in files]
    plotLearningCurves(curves, labelsList=labels if len(labels) else
    [getLabelFromFileName(file) for file in files],
                  title=title if title else getTitleFormFileName(files[0]), ylim=ylim)


def getTitleFormFileName(fileName: str) -> str:
    experiment = fileName.split('-')[0]
    realisation = fileName.split('-')[1].split('/')[0]
    return 'Eksperyment {}, realizacja {}'.format(experiment, realisation)


def getLabelFromFileName(fileName: str) -> str:
    experiment = fileName.split('-')[0]
    realisation = fileName.split('-')[1].split('/')[0]
    train = "Train" in fileName
    if len(fileName.split('-')[1].split('/')) > 2:
        return 'Eksperyment {}, realizacja {}{}  - zbiór {}'.format(experiment, realisation, 'A' if fileName.split('/')[1].split('/')[0] == '1' else 'B', 'Treningowy' if train  else 'Testowy')
    return 'Eksperyment {}, realizacja {} - zbiór {}'.format(experiment, realisation,'Treningowy' if train  else 'Walidacyjny')


if __name__ == "__main__":
    #%% E1
    processFiles(['1-1/run-tensorboard-tag-Train_Accuracy-top1.csv',
                  '1-2/run-tensorboard-tag-Train_Accuracy-top1.csv',
                  '1-3/run-7-tag-Train_Accuracy-top1.csv',
                  '1-1/run-tensorboard-tag-Val_Accuracy-top1.csv',
                  '1-2/run-tensorboard-tag-Val_Accuracy-top1.csv',
                  '1-3/run-7-tag-Val_Accuracy-top1.csv'
                  ],
                 directory='tensorboard_data',
                 title='Accuracy Top 1',
                 ylim=[75,100])
    processFiles(['1-1/run-tensorboard-tag-Train_Accuracy-top5.csv',
                  '1-2/run-tensorboard-tag-Train_Accuracy-top5.csv',
                  '1-3/run-7-tag-Train_Accuracy-top5.csv',
                  '1-1/run-tensorboard-tag-Val_Accuracy-top5.csv',
                  '1-2/run-tensorboard-tag-Val_Accuracy-top5.csv',
                  '1-3/run-7-tag-Val_Accuracy-top5.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[95, 100],
                 title='Accuracy Top 5')
    processFiles(['1-1/run-tensorboard-tag-Train_Mean-Avg-Precision.csv',
                  '1-2/run-tensorboard-tag-Train_Mean-Avg-Precision.csv',
                  '1-3/run-7-tag-Train_Mean-Avg-Precision.csv',
                  '1-1/run-tensorboard-tag-Val_Mean-Avg-Precision.csv',
                  '1-2/run-tensorboard-tag-Val_Mean-Avg-Precision.csv',
                  '1-3/run-7-tag-Val_Mean-Avg-Precision.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[0.6, 1],
                 title='Mean Avg Precision')
    processFiles(['1-1/run-tensorboard-tag-Train_Loss.csv',
                  '1-2/run-tensorboard-tag-Train_Loss.csv',
                  '1-3/run-7-tag-Train_Loss.csv',
                  '1-1/run-tensorboard-tag-Val_Loss.csv',
                  '1-2/run-tensorboard-tag-Val_Loss.csv',
                  '1-3/run-7-tag-Val_Loss.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[0, 1.5],
                 title='Loss')
    #%% E2
    processFiles(['2-1/run-tensorboard-tag-Train_Accuracy-top1.csv',
                  '2-2/run-tensorboard-tag-Train_Accuracy-top1.csv',
                  '2-3/run-7 (1)-tag-Train_Accuracy-top1.csv',
                  '2-1/run-tensorboard-tag-Val_Accuracy-top1.csv',
                  '2-2/run-tensorboard-tag-Val_Accuracy-top1.csv',
                  '2-3/run-7 (1)-tag-Val_Accuracy-top1.csv'
                  ],
                 directory='tensorboard_data',
                 title='Accuracy Top 1',
                 ylim=[75, 100])
    processFiles(['2-1/run-tensorboard-tag-Train_Accuracy-top5.csv',
                  '2-2/run-tensorboard-tag-Train_Accuracy-top5.csv',
                  '2-3/run-7 (1)-tag-Train_Accuracy-top5.csv',
                  '2-1/run-tensorboard-tag-Val_Accuracy-top5.csv',
                  '2-2/run-tensorboard-tag-Val_Accuracy-top5.csv',
                  '2-3/run-7 (1)-tag-Val_Accuracy-top5.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[95, 100],
                 title='Accuracy Top 5')
    processFiles(['2-1/run-tensorboard-tag-Train_Mean-Avg-Precision.csv',
                  '2-2/run-tensorboard-tag-Train_Mean-Avg-Precision.csv',
                  '2-3/run-7 (1)-tag-Train_Mean-Avg-Precision.csv',
                  '2-1/run-tensorboard-tag-Val_Mean-Avg-Precision.csv',
                  '2-2/run-tensorboard-tag-Val_Mean-Avg-Precision.csv',
                  '2-3/run-7 (1)-tag-Val_Mean-Avg-Precision.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[0.6, 1],
                 title='Mean Avg Precision')
    processFiles(['2-1/run-tensorboard-tag-Train_Loss.csv',
                  '2-2/run-tensorboard-tag-Train_Loss.csv',
                  '2-3/run-7 (1)-tag-Train_Loss.csv',
                  '2-1/run-tensorboard-tag-Val_Mean-Avg-Precision.csv',
                  '2-2/run-tensorboard-tag-Val_Mean-Avg-Precision.csv',
                  '2-3/run-7 (1)-tag-Val_Mean-Avg-Precision.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[0, 1.5],
                 title='Loss')
    #%%E3
    processFiles(['3-1/1/run-tensorboard-tag-first_Train_Accuracy-top1.csv',
                  '3-1/2/run-tensorboard-tag-second_Train_Accuracy-top1.csv',
                  '3-3/1/run-9-tag-first_Train_Accuracy-top1.csv',
                  '3-3/2/run-9-tag-second_Train_Accuracy-top1.csv',
                  '3-1/1/run-tensorboard-tag-first_Val_Accuracy-top1.csv',
                  '3-1/2/run-tensorboard-tag-second_Val_Accuracy-top1.csv',
                  '3-3/1/run-9-tag-first_Val_Accuracy-top1.csv',
                  '3-3/2/run-9-tag-second_Val_Accuracy-top1.csv'
                  ],
                 ylim=[75, 100],
                 directory='tensorboard_data',
                 title='Accuracy Top 1')
    processFiles(['3-1/1/run-tensorboard-tag-first_Train_Accuracy-top5.csv',
                  '3-1/2/run-tensorboard-tag-second_Train_Accuracy-top5.csv',
                  '3-3/1/run-9-tag-first_Train_Accuracy-top5.csv',
                  '3-3/2/run-9-tag-second_Train_Accuracy-top5.csv',
                  '3-1/1/run-tensorboard-tag-first_Val_Accuracy-top5.csv',
                  '3-1/2/run-tensorboard-tag-second_Val_Accuracy-top5.csv',
                  '3-3/1/run-9-tag-first_Val_Accuracy-top5.csv',
                  '3-3/2/run-9-tag-second_Val_Accuracy-top5.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[95, 100],
                 title='Accuracy Top 5')
    processFiles(['3-1/1/run-tensorboard-tag-first_Train_Mean-Avg-Precision.csv',
                  '3-1/2/run-tensorboard-tag-second_Train_Mean-Avg-Precision.csv',
                  '3-3/1/run-9-tag-first_Train_Mean-Avg-Precision.csv',
                  '3-3/2/run-9-tag-second_Train_Mean-Avg-Precision.csv',
                  '3-1/1/run-tensorboard-tag-first_Val_Mean-Avg-Precision.csv',
                  '3-1/2/run-tensorboard-tag-second_Val_Mean-Avg-Precision.csv',
                  '3-3/1/run-9-tag-first_Val_Mean-Avg-Precision.csv',
                  '3-3/2/run-9-tag-second_Val_Mean-Avg-Precision.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[0.4, 1],
                 title='Mean Avg Precision')
    processFiles(['3-1/1/run-tensorboard-tag-first_Train_Loss.csv',
                  '3-1/2/run-tensorboard-tag-second_Train_Loss.csv',
                  '3-3/1/run-9-tag-first_Train_Loss.csv',
                  '3-3/2/run-9-tag-second_Train_Loss.csv',
                  '3-1/1/run-tensorboard-tag-first_Val_Loss.csv',
                  '3-1/2/run-tensorboard-tag-second_Val_Loss.csv',
                  '3-3/1/run-9-tag-first_Val_Loss.csv',
                  '3-3/2/run-9-tag-second_Val_Loss.csv'
                  ],
                 directory='tensorboard_data',
                 ylim=[0, 4],
                 title='Loss')
    #%% E1 vs E2 vs E3
    processFiles(['1-3/run-7-tag-Val_Accuracy-top1.csv',
                  '2-3/run-7 (1)-tag-Val_Accuracy-top1.csv',
                  '3-3/1/run-9-tag-first_Val_Accuracy-top1.csv',
                  '3-3/2/run-9-tag-second_Val_Accuracy-top1.csv'
                  ],
                 ylim=[75, 90],
                 directory='tensorboard_data',
                 title='Najlepsze realizacje - Accuracy Top 1')
    processFiles(['1-3/run-7-tag-Val_Accuracy-top5.csv',
                  '2-3/run-7 (1)-tag-Val_Accuracy-top5.csv',
                  '3-3/1/run-9-tag-first_Val_Accuracy-top5.csv',
                  '3-3/2/run-9-tag-second_Val_Accuracy-top5.csv'],
                 directory='tensorboard_data',
                 ylim=[95, 100],
                 title='Najlepsze realizacje - Accuracy Top 5')
    processFiles(['1-3/run-7-tag-Val_Mean-Avg-Precision.csv',
                  '2-3/run-7 (1)-tag-Val_Mean-Avg-Precision.csv',
                  '3-3/1/run-9-tag-first_Val_Mean-Avg-Precision.csv',
                  '3-3/2/run-9-tag-second_Val_Mean-Avg-Precision.csv'],
                 directory='tensorboard_data',
                 ylim=[0.6, 0.9],
                 title='Najlepsze realizacje - Mean Avg Precision')
    processFiles(['1-3/run-7-tag-Val_Loss.csv',
                  '2-3/run-7 (1)-tag-Val_Loss.csv',
                  '3-3/1/run-9-tag-first_Val_Loss.csv',
                  '3-3/2/run-9-tag-second_Val_Loss.csv'],
                 directory='tensorboard_data',
                 ylim=[0, 1.5],
                 title='Najlepsze realizacje - Loss')
