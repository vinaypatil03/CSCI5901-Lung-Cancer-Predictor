import datetime
import pandas as pd


def get_date():
    date = datetime.datetime.now()
    return date.strftime("%d-%b-%Y")


def save_data(bs, ep, accuracy, total_time):
    df = pd.read_csv('metrics.csv')
    new_record = {'batch_size': bs, 'ephocs': ep, 'date': str(get_date()), 'accuracy': str(accuracy)[0:5],
                  'training_time': str(total_time)[0:5]}
    df = df.append(new_record, ignore_index=True)
    df.to_csv('metrics.csv', index=False)
    return True


def get_data_records():
    df = pd.read_csv('metrics.csv')
    return df.to_dict('records')
