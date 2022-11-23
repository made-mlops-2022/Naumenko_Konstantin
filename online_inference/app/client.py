import click
import requests


@click.command()
@click.argument('data_path')
@click.argument('prediction_path')
def predict(data_path, prediction_path):
    with open(data_path, 'rb') as f:
        url = 'http://127.0.0.1:8000/predict'
        file = {'file': f}
        r = requests.post(url, files=file)
    with open(prediction_path, 'wb') as f:
        f.write(r.content)


if __name__ == '__main__':
    predict()
