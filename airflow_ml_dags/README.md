# airflow-example


Команды, чтобы развернуть airflow, предварительно собрав контейнеры:
~~~
# для корректной работы с переменными, созданными из UI
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose build
docker-compose up
~~~

Пользователь/пароль: admin/admin

В variables необходимо добавить переменную model_name со значением даты необходимой модели, например, 2022-12-08

