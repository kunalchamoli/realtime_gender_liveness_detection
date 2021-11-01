FROM python:3.9-slim-bullseye

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV

#setting PATH: PATH is a list of directories which are searched for commands to run. 
#activate simply adds the virtualenv’s bin/ directory to the start of the list.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the application:
COPY myapp.py .
CMD ["python", "new_main.py"]