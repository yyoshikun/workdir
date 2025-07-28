FROM jupyter/datascience-notebook:python-3.10.9
USER root

RUN apt-get update
RUN apt-get install -y vim
RUN pip install --upgrade pip

RUN pip install jupyterlab torch==1.13.1 wandb openpyxl tqdm pandas matplotlib japanize-matplotlib

EXPOSE 8888

ENTRYPOINT [ "jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''" ]
CMD [ "--notebook-dir=/workdir" ]



#20250724へ変更
# FROM jupyter/datascience-notebook:python-3.10.9

#20230811_追加したけど動作しない

# FROM jupyter/datascience-notebook:python-3.9.13
# #FROM jupyter/datascience-notebook
# USER root

# RUN pip install jupyterlab==1.0
# RUN pip install japanize-matplotlib

# # decisionTree
# RUN pip install pydotplus
# RUN pip install dtreeviz
# RUN apt-get dist-upgrade
# RUN apt-get update
# RUN apt-get install -y build-essential graphviz-dev graphviz pkg-config
# RUN pip install graphviz

# RUN jupyter serverextension enable --py jupyterlab
# RUN jupyter labextension install jupyterlab_vim

# # nbextension
# RUN pip install jupyter-contrib-nbextensions
# RUN pip install jupyter-nbextensions-configurator
# RUN jupyter contrib nbextension install --user
# RUN jupyter nbextensions_configurator enable --user

# # CausalImpact
# RUN pip install pycausalimpact

# # Text
# RUN pip install wordcloud
# RUN pip install mecab-python3==0.996.5
# RUN apt-get update \
#     && apt-get install -y mecab \
#     && apt-get install -y mecab-ipadic \
#     && apt-get install -y libmecab-dev \
#     && apt-get install -y mecab-ipadic-utf8 \
#     && apt-get install -y swig \
#     && apt-get install -y file
# RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
#     && cd mecab-ipadic-neologd \
#     && bin/install-mecab-ipadic-neologd -n -y

# # SpreadSheet
# RUN pip install gspread
# RUN pip install oauth2client

# EXPOSE 8888
# CMD ["--notebook-dir=/workdir"]

#EXPOSE 10000
#CMD ["bash"]
#CMD ["jupyter lab --port 10000 --allow-root"]