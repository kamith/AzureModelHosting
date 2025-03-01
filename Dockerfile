FROM python:3.9

WORKDIR /code

COPY . .

RUN apt update
RUN apt install -y libgl1-mesa-glx

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["fastapi", "run", "main.py", "--port", "80"]